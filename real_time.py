# FILE: real_time_inference.py

import torch
import numpy as np
import sounddevice as sd
from pyctcdecode import build_ctcdecoder
import torchaudio
from collections import OrderedDict
import queue
import webrtcvad
import sys

# Your custom modules
from model import SpeechRecognitionModel

# --- Basic Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = 'model_rec.pth'
SAMPLERATE = 16000
CHUNK_DURATION_MS = 30  # VAD requires 10, 20, or 30 ms chunks
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION_MS / 1000) # frames per buffer

# --- Load Labels ---
LABELS = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'", "_"]
BLANK_ID = LABELS.index("_")

# Thread-safe queue for audio data
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called by sounddevice for each new audio chunk."""
    if status:
        print(status, file=sys.stderr)
    # The VAD needs bytes
    audio_queue.put(indata.tobytes())

def main():
    # --- 1. Load the Model ---
    print("Loading model...")
    model = SpeechRecognitionModel(
        n_mels=64, n_class=len(LABELS), n_hidden=512, n_layers=2
    ).to(DEVICE).eval()
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[10:] if k.startswith('_orig_mod.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return
    print("Model loaded successfully.")

    # --- 2. Load the Decoder ---
    print("Loading CTC Beam Search Decoder...")
    try:
        kenlm_model_path = 'lm.bin'
        decoder = build_ctcdecoder(
            labels=[l for l in LABELS if l != '_'],
            kenlm_model_path=kenlm_model_path,
            alpha=0.5,
            beta=2.0
        )
    except FileNotFoundError:
        print("Warning: KenLM model 'lm.bin' not found. Decoding will be less accurate.")
        decoder = build_ctcdecoder(labels=[l for l in LABELS if l != '_'])
    print("Decoder loaded successfully.")
    
    # --- 3. Set up Feature Extraction ---
    feature_transform = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLERATE, n_fft=512, win_length=400, hop_length=160,
            n_mels=64, f_min=20, f_max=7600, window_fn=torch.hann_window, power=1.0
        ),
        torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80),
    ).to(DEVICE)

    def cmvn(tensor, mean=-14.0, std=15.0):
        return (tensor - mean) / std

    # --- 4. Setup VAD and Real-Time Loop ---
    vad = webrtcvad.Vad(3)  # Set aggressiveness from 0 (least) to 3 (most)
    
    print("\n--- Starting Real-Time Transcription ---")
    print("Speak into the microphone (press Ctrl+C to stop)...")
    
    # Start the audio stream in a non-blocking way
    stream = sd.InputStream(
        samplerate=SAMPLERATE,
        blocksize=CHUNK_SIZE,
        device=None,
        channels=1,
        dtype='int16', # VAD requires 16-bit PCM
        callback=audio_callback
    )
    stream.start()

    speech_segment = bytearray()
    is_speaking = False
    silence_chunks = 0
    padding_chunks = 4 # Number of padding chunks to add to the end of speech

    try:
        while True:
            chunk = audio_queue.get()
            is_speech = vad.is_speech(chunk, SAMPLERATE)

            if is_speaking:
                if is_speech:
                    speech_segment.extend(chunk)
                    silence_chunks = 0 # Reset silence counter
                else:
                    # If not speech, start counting silence chunks
                    silence_chunks += 1
                    speech_segment.extend(chunk) # Add some padding
                    # If silence is long enough, we have the end of an utterance
                    if silence_chunks > padding_chunks:
                        is_speaking = False
                        
                        # --- Process the completed utterance ---
                        # Convert bytearray to numpy array and then to tensor
                        audio_np = np.frombuffer(speech_segment, dtype=np.int16).astype(np.float32) / 32768.0
                        waveform = torch.from_numpy(audio_np).float().to(DEVICE)
                        
                        if len(waveform) > 0:
                            with torch.no_grad():
                                features = feature_transform(waveform.unsqueeze(0))
                                features = features.transpose(1, 2)
                                features = cmvn(features)
                                log_probs = model(features)
                                log_probs_numpy = log_probs.cpu().numpy()

                                # Use the verified decode_beams function
                                beam_results = decoder.decode_beams(log_probs_numpy[0], beam_width=100)
                                transcription = beam_results[0][0]
                                
                                # Overwrite the line with the final transcription
                                sys.stdout.write("\r" + " " * 80) # Clear line
                                sys.stdout.flush()
                                print(f"\rFinal: {transcription}")
                        
                        # Reset for the next utterance
                        speech_segment.clear()

            elif is_speech:
                # Start of a new utterance
                is_speaking = True
                speech_segment.extend(chunk)
                sys.stdout.write("\rSpeaking...")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n--- Stopping Transcription ---")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()