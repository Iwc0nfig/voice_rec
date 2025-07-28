# test_model.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict, Counter
import os
import jiwer  # A popular library for calculating Word Error Rate

# Import your custom modules
from my_dataset import LibriSpeechDataset, collate_fn # Assuming you have a test CSV
from model import SpeechRecognitionModel

# --- Import and configure the decoder from my_decoder.py ---
from pyctcdecode import build_ctcdecoder

# --- Basic Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED_MODEL_PATH = 'model_rec.pth'
TEST_CSV_PATH = 'val.csv' # IMPORTANT: Create a CSV for your test data
BATCH_SIZE = 16 # You can adjust this based on your VRAM

print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')


# --- 1. Load Labels (same as in your other scripts) ---
labels = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'", "_"]
blank_char = "_"
blank_id = labels.index(blank_char)


# --- 2. Load the Fine-Tuned Model ---
print(f"Loading fine-tuned model from '{FINETUNED_MODEL_PATH}'...")

# Initialize the model structure
model = SpeechRecognitionModel(
    n_mfcc=13,
    n_class=len(labels),
    n_hidden=256,
    n_layers=2,
    dropout=0.0 # Dropout is typically disabled for inference
)

if not os.path.exists(FINETUNED_MODEL_PATH):
    raise FileNotFoundError(f"Error: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'")

# Load the state dict, handling the torch.compile() prefix
state_dict = torch.load(FINETUNED_MODEL_PATH, map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('_orig_mod.'):
        name = k[10:] # remove `_orig_mod.`
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
print("Model loaded and weights corrected successfully.")

model.to(device)
model.eval() # Set the model to evaluation mode

# Compile the model for faster inference (optional but recommended)
# model = torch.compile(model)


# --- 3. Set up the CTC Beam Search Decoder ---
print("Setting up the CTC Beam Search Decoder...")
# Load unigrams from your corpus for the decoder
try:
    word_counts = Counter()
    with open('corpus.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().upper().split() # Ensure words are uppercase to match labels
            word_counts.update(words)
    unigrams = list(word_counts.keys())
    print(f"Loaded {len(unigrams)} unigrams from corpus.txt")
except FileNotFoundError:
    print("Warning: corpus.txt not found. Decoder will work without unigrams, but performance may be lower.")
    unigrams = None

# Decoder hyperparameters
kenlm_model_path = 'lm.bin'
alpha = 0.7  # Language model weight
beta = 2.0   # Word insertion bonus

try:
    decoder = build_ctcdecoder(
        labels=[l for l in labels if l != '_'], # Decoder labels should not include the blank character
        kenlm_model_path=kenlm_model_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta
    )
    print("CTC decoder loaded successfully.")
except Exception as e:
    print(f"Error loading decoder: {e}. Exiting.")
    exit()


# --- 4. Load the Test Dataset ---
print(f"Loading test data from '{TEST_CSV_PATH}'...")
test_dataset = LibriSpeechDataset(
    csv_path=TEST_CSV_PATH,
    labels=labels
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle for testing
    collate_fn=collate_fn
)


# --- 5. The Main Inference and Evaluation Loop ---
ground_truths = []
predictions = []

print("\n--- Starting Inference on Test Set ---")
# Disable gradient calculations for inference
with torch.no_grad():
    for i, (mfccs, transcripts, mfccs_lengths, _) in enumerate(test_loader):
        mfccs = mfccs.to(device)

        # Get model output (log probabilities)
        log_probs = model(mfccs)

        # The decoder expects numpy arrays on the CPU
        log_probs_numpy = log_probs.cpu().numpy()

        # Loop through each item in the batch
        for j in range(log_probs_numpy.shape[0]):
            single_log_probs = log_probs_numpy[j][:mfccs_lengths[j]] # Use actual length
            
            # Decode the output
            # The result is a list of beams. The first one is the best.
            beam_results = decoder.decode_beams(single_log_probs, beam_width=100)
            
            # Extract the transcription from the best beam
            if beam_results:
                best_beam = beam_results[0]
                # The first element of the beam tuple is the transcription
                decoded_transcription, _, _, _, _ = best_beam
            else:
                decoded_transcription = ""

            # Store ground truth and prediction for WER calculation
            ground_truth_tensor = transcripts[j]
            ground_truth = "".join([labels[i] for i in ground_truth_tensor.tolist() if labels[i] != "_"])

            
            ground_truths.append(ground_truth)
            predictions.append(decoded_transcription)
            

            

            # Print progress
            print(f"Sample {i*BATCH_SIZE + j + 1}/{len(test_dataset)}")
            #print(f"  GT:    '{ground_truth}'")
            #print(f"  Pred:  '{decoded_transcription}'\n")


# --- 6. Calculate and Print Final Results ---
print("--- Evaluation Complete ---")
wer = jiwer.wer(ground_truths, predictions)
cer = jiwer.cer(ground_truths, predictions)

print(f"Final Word Error Rate (WER): {wer * 100:.2f}%")
print(f"Final Character Error Rate (CER): {cer * 100:.2f}%")

# You can also save the results to a file
with open('test_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Word Error Rate (WER): {wer * 100:.2f}%\n")
    f.write(f"Character Error Rate (CER): {cer * 100:.2f}%\n\n")
    for gt, pred in zip(ground_truths, predictions):
        f.write(f"GT:   {gt}\n")
        f.write(f"Pred: {pred}\n\n")

print("\nTest results saved to 'test_results.txt'")