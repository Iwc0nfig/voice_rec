import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import logging
import json
import os
from collections import OrderedDict

# --- Basic setup (same as train.py) ---
logging.basicConfig(
    filename='finetune.log', # Log to a new file
    level= logging.INFO,
    format='%(message)s'
)

# Import your custom modules
from my_dataset import LibriSpeechDataset, collate_fn
from model import SpeechRecognitionModel

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.set_float32_matmul_precision('high')

labels = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'", "_"]
blank_char = "_"
blank_id = labels.index(blank_char)

# --- Fine-Tuning Hyperparameters ---
fine_tune_lr = 1e-4  # Use a lower learning rate for fine-tuning
batch_size = 64
fine_tune_epochs = 10 # Number of additional epochs to train
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- Model Paths ---
PRETRAINED_MODEL_PATH = 'model_rec.pth'
FINETUNED_MODEL_PATH = 'model_rec_finetuned.pth'


def validate_model(model, val_loader, criterion, device):
    """Calculate validation loss"""
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for mfccs, labels, mfccs_lengths, labels_length in val_loader:
            mfccs = mfccs.to(device)
            labels = labels.to(device)
            
            log_probs = model(mfccs)
            log_probs = log_probs.permute(1, 0, 2)
            
            val_loss = criterion(log_probs, labels, mfccs_lengths, labels_length)
            total_val_loss += val_loss.item()
            num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    return avg_val_loss

# --- Dataset and DataLoader (same as train.py) ---
train_dataset = LibriSpeechDataset(
    csv_path='metadata.csv',
    labels=labels
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = LibriSpeechDataset(
    'val.csv',
    labels=labels
)

val_loader =DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


# --- Model Definition ---
model = SpeechRecognitionModel(
    n_mfcc=40,
    n_class=len(labels),
    n_hidden=256,
    n_layers=2,
    dropout=0.2
)

# --- LOAD PRE-TRAINED WEIGHTS ---
if os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading pre-trained model from '{PRETRAINED_MODEL_PATH}' for fine-tuning...")

    # Load the state dict, addressing the security warning
    # This also handles the case where the model was compiled
    state_dict = torch.load(PRETRAINED_MODEL_PATH, weights_only=True)

    # --- FIX for torch.compile() ---
    # Create a new state dictionary without the '_orig_mod.' prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            name = k[10:] # remove `_orig_mod.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    # Load the corrected state dict
    model.load_state_dict(new_state_dict)
    print("Pre-trained weights loaded and corrected successfully.")
else:
    raise FileNotFoundError(f"Error: Pre-trained model not found at '{PRETRAINED_MODEL_PATH}'")

model.to(device)


# --- Define new Optimizer and Scheduler for Fine-Tuning ---
criterion = CTCLoss(blank=blank_id, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(),lr=fine_tune_lr , weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=fine_tune_epochs,
    eta_min=fine_tune_lr * 0.5
)

# Compile the model for performance AFTER loading the state dict
model = torch.compile(model)


# --- Fine-Tuning Loop ---
print("--- Starting Fine-Tuning ---")
for epoch in range(fine_tune_epochs):
    model.train()
    total_loss = 0

    for i, (mfccs, labels, mfccs_lengths, labels_length) in enumerate(train_loader):
        mfccs = mfccs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        log_probs = model(mfccs)
        log_probs = log_probs.permute(1,0,2)

        loss = criterion(log_probs,labels,mfccs_lengths,labels_length)
        loss.backward()

        
            
        optimizer.step()

        total_loss += loss
        current_lr = optimizer.param_groups[0]['lr']

        log_entry = {
            'epoch': epoch + 1,
            'step': i + 1,
            'total_steps': len(train_loader),
            'loss': round(loss.item(), 6),
            'lr': round(current_lr, 6)
        }
        logging.info(json.dumps(log_entry))    

        


        if(i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{fine_tune_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Learning rate = {current_lr:.6f} ')


        avg_train_loss = total_loss / len(train_loader)

    avg_val_loss = validate_model(model, val_loader, criterion, device)

    epoch_summary = {
        'epoch': epoch + 1,
        'avg_train_loss': round(avg_train_loss.item(), 6),
        'avg_val_loss': round(avg_val_loss, 6)
    }
    logging.info(json.dumps(epoch_summary))


    print(f'End of Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}')
    scheduler.step()


# --- Save the Fine-Tuned Model to a NEW file ---
# The state_dict will be saved with the '_orig_mod.' prefix again, which is fine.
# The loading logic at the top can handle it.
torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
print(f"Fine-tuning complete. Model saved to '{FINETUNED_MODEL_PATH}'")