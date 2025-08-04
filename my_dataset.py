# Fixed dataset.py with robust filtering for CTC loss and SpecAugment

from torch.nn.utils.rnn import pad_sequence
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import random

def collate_fn(batch):
    """
    Filters out failed samples (None), then pads the rest.
    
    Args:
        batch: A list of tuples (features, label) which might contain Nones.
        
    Returns:
        A tuple of padded Tensors or (None, None, None, None) if batch is empty.
    """
    # 1. Filter out None values from failed __getitem__ calls
    batch = [b for b in batch if b is not None]
    if not batch:
        # If the whole batch consists of bad data, return None
        return None, None, None, None

    # 2. Proceed with collation as before
    features, labels = zip(*batch)
    feature_lengths = torch.tensor([len(f) for f in features])
    label_lengths = torch.tensor([len(l) for l in labels])

    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_features, padded_labels, feature_lengths, label_lengths


class LibriSpeechDataset(Dataset):
    """
    A PyTorch Dataset for LibriSpeech, optimized for CTC-based speech recognition.
    Includes filtering for data that violates the CTC loss constraints and data augmentation.
    """
    def __init__(self, csv_path, labels, sample_rate=16000, augmentation=False, noise_prob=0.5, 
                 noise_factor_range=(0.005, 0.02), volume=False, spec_augment=False,
                 time_mask_param=35, time_mask_num=2, freq_mask=False, freq_mask_param=15, freq_mask_num=2):
        self.df = pd.read_csv(csv_path)
        self.labels_map = {label: i for i, label in enumerate(labels)}
        self.sample_rate = sample_rate
        self.n_mels = 64
        self.augmentation = augmentation
        self.noise_prob = noise_prob
        self.noise_factor_range = noise_factor_range
        self.volume = volume
        self.spec_augment = spec_augment
        self.freq_mask = freq_mask  # Option to enable/disable frequency masking

        self.feature_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16_000,
                n_fft=512,              # 32 ms window → sharper time resolution
                win_length=400,         # 25 ms (common in ASR)
                hop_length=160,         # 10 ms stride
                n_mels=64,              # matches many CNN front-ends
                f_min=20,
                f_max=7_600,            # Nyquist – 400 Hz guard
                window_fn=torch.hann_window,
                power=1.0               # magnitude → log-Mel-**amplitude**
            ),
            torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80),
        )
        
        # SpecAugment parameters
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        
        # Initialize SpecAugment transforms
        if self.spec_augment:
            if self.freq_mask:
                self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)

        self.cnn_downsample_factor = 1

        print(f"LibriSpeech Dataset initialized:")
        print(f" - Sample rate: {self.sample_rate}Hz")
        print(f" - CNN Downsample Factor: {self.cnn_downsample_factor}")
        print(f" - Total samples: {len(self.df)}")
        print(f" - Augmentation : {self.augmentation}")
        if self.augmentation:
            print(f" - Noise probability: {self.noise_prob}")
            print(f" - Noise factor range: {self.noise_factor_range}")
        if self.volume:
            print(f" - Modify the volume of the data.")
        if self.spec_augment:
            print(f" - SpecAugment enabled (optimized for 1D CNN):")
            print(f"   - Time masks: {self.time_mask_num} masks, max width {self.time_mask_param}")
            if self.freq_mask:
                print(f"   - Frequency masks: {self.freq_mask_num} masks, max width {self.freq_mask_param}")
            else:
                print(f"   - Frequency masking disabled (recommended for 1D CNNs)")

    def cmvn(self, tensor, mean=-14.0, std=15.0):
        return (tensor - mean) / std
    
    def add_noise(self, waveform):
        if not self.augmentation or random.random() > self.noise_prob:
            return waveform
        
        noise = torch.randn_like(waveform)
        
        # Random noise factor within the specified range
        noise_factor = random.uniform(*self.noise_factor_range)
        
        # Scale noise relative to signal power
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Scale noise to achieve desired SNR
        if noise_power > 0:
            noise = noise * torch.sqrt(signal_power * noise_factor / noise_power)
        
        return waveform + noise
    
    def volume_perturbation(self, waveform, gain_range=(0.8, 1.2)):
        """
        Apply random volume changes.
        
        Args:
            waveform: Input audio waveform tensor
            gain_range: Range of gain factors
            
        Returns:
            Volume-adjusted waveform tensor
        """
        if not self.augmentation or random.random() > 0.3:  # Lower probability for volume changes
            return waveform
            
        gain = random.uniform(*gain_range)
        return waveform * gain
    
    def apply_spec_augment(self, spectrogram):
        """
        Apply SpecAugment to the spectrogram.
        Optimized for 1D CNNs - frequency masking is optional and disabled by default.
        
        Args:
            spectrogram: Input spectrogram tensor of shape (freq, time)
            
        Returns:
            Augmented spectrogram tensor
        """
        if not self.spec_augment:
            return spectrogram
            
        # Add batch dimension for torchaudio transforms if needed
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False
            
        # Apply frequency masking only if enabled (not recommended for 1D CNN)
        if self.freq_mask:
            for _ in range(self.freq_mask_num):
                spectrogram = self.freq_masking(spectrogram)
            
        # Apply time masking (recommended for 1D CNN)
        for _ in range(self.time_mask_num):
            spectrogram = self.time_masking(spectrogram)
            
        # Remove batch dimension if we added it
        if squeeze_after:
            spectrogram = spectrogram.squeeze(0)
            
        return spectrogram
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['filepath']
        text = self.df.iloc[idx]['text']

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Apply waveform-level augmentations
            if self.augmentation:
                waveform = self.add_noise(waveform)
            if self.volume:
                waveform = self.volume_perturbation(waveform)

            # Extract features
            features = self.feature_transform(waveform).squeeze(0)  # Shape: (n_mels, time)
            
            # Apply SpecAugment before transposing and CMVN
            features = self.apply_spec_augment(features)
            
            # Transpose to (time, freq) and apply CMVN
            features = features.transpose(0, 1)  # Now (time, freq)
            features = self.cmvn(features)  # global CMVN

            label_indices = [self.labels_map[char] for char in text if char in self.labels_map]
            if not label_indices:
                print("No label_indices")
                return None

            label = torch.tensor(label_indices, dtype=torch.long)

            # CTC constraint check
            output_length = features.shape[0] // self.cnn_downsample_factor
            if output_length < len(label):
                print(f"Warning: Skipping {audio_path}. Downsampled audio ({output_length}) is shorter than label ({len(label)}).")
                return None

            return features, label

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None