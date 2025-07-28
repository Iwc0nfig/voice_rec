import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import logging
import json
import math

#logging
logging.basicConfig(
    filename='train.log',
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

max_lr = 3e-4
min_lr = max_lr *0.1
batch_size = 64
epochs = 30
warmup_steps = 4
max_steps = 30
#device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def get_lr(it):
    # 1) linear warmup for warmup_steps steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) cosine decay to min_lr after warmup_steps
    elif it < max_steps:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine decay
        return min_lr + coeff * (max_lr - min_lr)
    # 3) at max_steps and beyond, return min_lr
    else:
        return min_lr# linear decay from min_lr to max_lr


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

train_dataset = LibriSpeechDataset(
    csv_path='metadata.csv',
    labels=labels
)



train_loader =DataLoader(
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

model = SpeechRecognitionModel(
    n_mfcc=13,
    n_class=len(labels),
    n_hidden=256,
    n_layers=2,
    dropout=0.2
).to(device)

criterion = CTCLoss(blank=blank_id,zero_infinity=True)
optimizer = optim.AdamW(model.parameters(),lr=max_lr , weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=epochs,
#     eta_min=learning_rate * 0.1
# )

model = torch.compile(model)


for epoch in range(epochs):
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

        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
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
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Learning rate = {current_lr:.6f} ')


        avg_train_loss = total_loss / len(train_loader)

    avg_val_loss = validate_model(model, val_loader, criterion, device)

    epoch_summary = {
        'epoch': epoch + 1,
        'avg_train_loss': round(avg_train_loss.item(), 6),
        'avg_val_loss': round(avg_val_loss, 6)
    }
    logging.info(json.dumps(epoch_summary))


    print(f'End of Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}')
    # scheduler.step()


torch.save(model.state_dict(), 'model_rec.pth')