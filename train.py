import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import logging
import json
import warnings
import math
import time 
# Import your custom modules
from my_dataset import LibriSpeechDataset, collate_fn
from model import SpeechRecognitionModel
from torch.optim.lr_scheduler import LinearLR, SequentialLR ,CosineAnnealingLR


warnings.filterwarnings("ignore", category=UserWarning)
print("WARNINGS will not been shown .")
print("Enable warning for better debuggin")
#logging
logging.basicConfig(
    filename='train_v2.log',
    level= logging.INFO,
    format='%(message)s'
)



#optimize setting for torch cuda
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

labels = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'", "_"]
blank_char = "_"
blank_id = labels.index(blank_char)

max_lr = 3e-4
min_lr = 1e-5
batch_size = 256
epochs = 40
warmup_steps = 4  # A more reasonable number of steps for warmup
max_steps = 40   # Total steps for cosine decay
compiled = False
save_model_per_epoch = 5


def time_for_epoch(t1,t2):
    tmp = t2-t1
    minuites = int(tmp // 60)
    sec = int(tmp%60)
    return f"{minuites}:{sec}"

def print_model_size(model: SpeechRecognitionModel):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_mb = total_size_bytes / (1024 * 1024)  # Convert bytes to MB

    print(f"The model has {total_params:,} parameters")  # comma as thousands separator
    print(f"The size of the model is {total_size_mb:.2f} MB")

    
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Batch size  : {batch_size}")




def validate_model(model, val_loader, criterion, device):
    """Calculate validation loss"""
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for features, labels, features_length, labels_length in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            features_length = features_length.to(device)
            
            
            log_probs = model(features)
            log_probs = log_probs.permute(1, 0, 2)

            
            val_loss = criterion(log_probs,labels,features_length,labels_length)

            total_val_loss += val_loss.item()
            num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    return avg_val_loss



train_dataset = LibriSpeechDataset(
    csv_path='metadata.csv',
    labels=labels,
    augmentation=True,
    volume=True,
    spec_augment=True,
    freq_mask=False
)    



train_loader =DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=16
)

val_dataset = LibriSpeechDataset(
    'val.csv',
    labels=labels,
            
)
val_loader =DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=16
)

model = SpeechRecognitionModel(
    n_mels=64,
    n_class=len(labels),
    n_hidden=512,
    n_layers=2,
    dropout=0.3,
    cnn_dropout=0.3,
).to(device)

criterion = CTCLoss(blank=blank_id,zero_infinity=True)
optimizer = optim.AdamW(model.parameters(),lr=max_lr , weight_decay=3e-4)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, 
                        schedulers=[warmup_scheduler, cosine_scheduler], 
                        milestones=[warmup_steps])

print(" ---- Learning Rate Config ")
print(f" - Max Learning Rate : {max_lr}")
print(f" - Min Learning Rate : {min_lr}")
print(f" - Warmpup steps : {warmup_steps}")
print(f" - Cosine Scheduler ")


try:
    model = torch.compile(model)    
    print('Model has been compiled')
except Exception  as e:

    print("Model has NOT been compiled")
    print("-"*50)
    print(e)
    print("-"*50)
    move_on = input("Do you want to continue without compile the model (y/n): ")
    if move_on.strip() != "y":
        import sys 
        sys.exit()

print_model_size(model)



def main():
    print(f"Saving after {save_model_per_epoch} epochs ")
    for epoch in range(epochs):
        t1 = time.time()
        model.train()
        total_loss = 0

        for i, (features, labels, features_length, labels_length) in enumerate(train_loader):
            features  = features.to(device)
            labels = labels.to(device)
            features_length = features_length.to(device)
    
            optimizer.zero_grad()

            # lr = get_lr(epoch)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            log_probs = model(features)
            log_probs = log_probs.permute(1,0,2)


            loss = criterion(log_probs,labels,features_length,labels_length)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected at epoch {epoch+1}, step {i+1}. Skipping step.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

            
                
            optimizer.step()
        

            total_loss += loss
            current_lr = optimizer.param_groups[0]['lr']

            log_entry = {
                'epoch': epoch + 1,
                'step': i + 1,
                'total_steps': len(train_loader),
                'loss': round(loss.item(), 5),
                'lr': round(current_lr, 7)
            }
            logging.info(json.dumps(log_entry))    

            

            if(i+1) % 40 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.5f}, Learning rate = {current_lr:.7f} ')


            avg_train_loss = total_loss / len(train_loader)
        scheduler.step()
        

        t2 = time.time()
        avg_val_loss = validate_model(model, val_loader, criterion, device)
        t3 = time.time()
        epoch_summary = {
            'epoch': epoch + 1,
            'avg_train_loss': round(avg_train_loss.item(), 6),
            'avg_val_loss': round(avg_val_loss, 6)
        }
        logging.info(json.dumps(epoch_summary))


        print(f'End of Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f} , time : {time_for_epoch(t1,t2)} , val_time : {time_for_epoch(t2,t3)}')
        


        if (epoch+1)%save_model_per_epoch == 0:
            model_name = f"model_rec_{epoch+1}.pth"
            print(f"Saving model at epoch : {epoch} and name : {model_name  }")
            torch.save(model.state_dict(), model_name)



    torch.save(model.state_dict(), 'model_rec.pth')
    

if __name__ == "__main__":
    print("-"*50)
    main()