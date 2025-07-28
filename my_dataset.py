# Fixed dataset.py with LibriSpeech train-clean-100 optimized configuration

from torch.nn.utils.rnn import pad_sequence
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import string


def create_librispeech_labels():
    """
    Create character labels for LibriSpeech dataset.
    LibriSpeech uses uppercase letters, space, and apostrophe.
    """
    # LibriSpeech character set: A-Z, space, apostrophe
    chars = list(string.ascii_uppercase) + [' ', "'"]
    
    # Add special tokens
    labels = ['<blank>'] + chars  # CTC blank token at index 0
    
    print(f"Created {len(labels)} labels: {labels}")
    return labels


def collate_fn(batch):
    """
    Pads data in a batch.
    
    Args:
        batch: A list of tuples (mfccs, label).
        
    Returns:
        A tuple of padded Tensors: (padded_mfccs, padded_labels, mfcc_lengths, label_lengths).
    """
    # Unzip the batch
    mfccs, labels = zip(*batch)

    # Get the lengths before padding
    mfcc_lengths = torch.tensor([len(m) for m in mfccs])
    label_lengths = torch.tensor([len(l) for l in labels])

    # Pad the sequences
    padded_mfccs = pad_sequence(mfccs, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_mfccs, padded_labels, mfcc_lengths, label_lengths


class LibriSpeechDataset(Dataset):
    """
    A PyTorch Dataset for LibriSpeech train-clean-100, optimized for speech recognition.
    """
    def __init__(self, csv_path, labels, sample_rate=16000, n_mfcc=13):
        self.df = pd.read_csv(csv_path)
        self.labels_map = {label: i for i, label in enumerate(labels)}
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        # LibriSpeech-optimized parameters
        # These parameters are proven to work well with LibriSpeech
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 2048,        # 128ms window at 16kHz
                'hop_length': 512,    # 32ms hop (75% overlap)
                'n_mels': 80,         # Good balance for speech
                'f_min': 0.0,
                'f_max': 8000.0,      # Speech content mostly below 8kHz
                'window_fn': torch.hann_window,
                'power': 2.0,
                'normalized': False
            }
        )
        
        print(f"LibriSpeech Dataset initialized:")
        print(f"  - Sample rate: {self.sample_rate}Hz")
        print(f"  - N-FFT: 2048 (128ms window)")
        print(f"  - Hop length: 512 (32ms)")
        print(f"  - Mel filters: 80")
        print(f"  - MFCC coefficients: {n_mfcc}")
        print(f"  - Total samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['filepath']
        text = self.df.iloc[idx]['text']

        try:
            # 1. Load audio (LibriSpeech is already 16kHz mono)
            waveform, sr = torchaudio.load(audio_path)
            
            # LibriSpeech files should already be 16kHz, but double-check
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Ensure mono (LibriSpeech should already be mono)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 2. Extract MFCC features
            # Output shape: (n_mfcc, time) -> transpose to (time, n_mfcc)
            mfccs = self.mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            
            # 3. Normalize MFCCs (recommended for LibriSpeech)
            mfccs = (mfccs - mfccs.mean(dim=0)) / (mfccs.std(dim=0) + 1e-8)

            # 4. Encode text to character indices
            # LibriSpeech transcripts are already uppercase and clean
            label_indices = []
            for c in text:  # Don't convert to upper since LibriSpeech is already processed
                if c in self.labels_map:
                    label_indices.append(self.labels_map[c])
                elif c == ' ':
                    # Handle space character if not in labels_map
                    if ' ' in self.labels_map:
                        label_indices.append(self.labels_map[' '])
                    else:
                        # Skip spaces or add a space token to your labels
                        continue
                else:
                    # For debugging: print unknown characters
                    if len(label_indices) == 0:  # Only print once per sample
                        print(f"Unknown character '{c}' in text: '{text[:50]}...'")
            
            if len(label_indices) == 0:
                # Fallback for empty labels
                label_indices = [0]  # Assuming 0 is blank/padding token
                
            label = torch.tensor(label_indices, dtype=torch.long)
            
            return mfccs, label
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            print(f"  Text: {text}")
            # Return minimal valid tensors
            return torch.zeros((1, self.n_mfcc)), torch.tensor([0], dtype=torch.long)

