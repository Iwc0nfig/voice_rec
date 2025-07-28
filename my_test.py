import torch
import sounddevice as sd
from model import SpeechRecognitionModel
import torchaudio
import torchaudio.transforms as T
import numpy as np




SAMPLE_RATE = 16000
BUFFER_SECONDS = 3  # Record 3 seconds at a time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

labels = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'", "_"]
blank_char = "_"
blank_id = labels.index(blank_char)
LM_PATH = 'lm.bin'


model = SpeechRecognitionModel(
    n_mfcc=13,
    n_class=len(labels),
    n_hidden=256,
    n_layers=2)  

model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval().to(DEVICE)


melkwargs = {
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 80,
    'f_min': 0.0,
    'f_max': 8000.0,
    'window_fn': torch.hann_window,
    'power': 2.0,
    'normalized': False
}


mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=13,
    melkwargs=melkwargs
)

decoder = BeamSearchDecoderCTC(
    vocab=vocab,
    kenlm_model_path=LM_PATH,
    beam_width=100,
    alpha=0.6,
    beta=1.0,
    blank_token=labels[blank_id]
)

def transcribe(audio_np):
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)  # [1, T]
    mfcc = mfcc_transform(waveform)
    mfcc = mfcc.transpose(1, 2).to(DEVICE)  # [1, T, 13]
    
    with torch.no_grad():
        log_probs = model(mfcc)  # [1, T, C]
    
    probs_np = log_probs.squeeze(0).cpu().numpy()
    text = decoder.decode(probs_np)
    return text

