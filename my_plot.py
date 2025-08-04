import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any




def read_training_log(log_file_path: str) -> List[Dict[str, Any]]:
    """
    Read and parse training log file containing JSON entries.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        List of parsed log entries as dictionaries
    """
    log_entries = []
    
    try:
        with open(log_file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        entry = json.loads(line)
                        log_entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num}: {line}")
                        print(f"JSON Error: {e}")
                        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []
    df = pd.DataFrame(log_entries)
    return df



def plot_training_metrics(df: pd.DataFrame):
    
    
    # Create subplots
    fig , (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot loss
    ax1.plot(df.index, df['loss'], 'b-', linewidth=1.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    length = int(len(df)*0.7)
    ends = df[length:]
    
    ax2.plot(ends.index, ends['loss'], 'b-', linewidth=1.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss last 9 epochs')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    




# Example usage
if __name__ == "__main__":
    log_file = 'train.log'
    fine_tune = 'finetune.log'
    
    entities = read_training_log(log_file)
    if entities['avg_train_loss'] is not None:
        print(len(entities['avg_train_loss']))
        print(len(entities['avg_val_loss']))
        epoch = -29
        for train , val in zip(entities['avg_train_loss'],entities['avg_val_loss']):
            if train > 0 and val >0:
                print(f"epoch = {epoch} , train_loss = {train} , val_loss = {val}")
                epoch +=1
    #plot_training_metrics(entities)
    #plot_training_metrics(fine_tune)
    #plt.show()
    
    
