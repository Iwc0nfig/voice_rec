
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict, Counter
import os
import jiwer
import itertools
from tqdm import tqdm
import json
import time

# Import your custom modules
from my_dataset import LibriSpeechDataset, collate_fn
from model import SpeechRecognitionModel
from pyctcdecode import build_ctcdecoder

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED_MODEL_PATH = 'model_rec.pth'
TEST_CSV_PATH = 'val.csv'
BATCH_SIZE = 16

# Optimization parameters
ALPHA_RANGE = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
BETA_RANGE = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
BEAM_WIDTH = 100

# For quick testing, you can use a subset of your data
USE_SUBSET = True  # Set to False to use full dataset
SUBSET_SIZE = 100  # Number of samples to use for optimization

print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')

# --- Load Labels ---
labels = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'", "_"]
blank_char = "_"
blank_id = labels.index(blank_char)

# --- Load Model ---
print(f"Loading fine-tuned model from '{FINETUNED_MODEL_PATH}'...")
model = SpeechRecognitionModel(
    n_mels=64,
    n_class=len(labels),
    n_hidden=256,
    n_layers=2,
    dropout=0.3,
    cnn_dropout=0.3,
)

if not os.path.exists(FINETUNED_MODEL_PATH):
    raise FileNotFoundError(f"Error: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'")

state_dict = torch.load(FINETUNED_MODEL_PATH, map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('_orig_mod.'):
        name = k[10:]
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

print("Model loaded successfully.")

# --- Load Unigrams ---
print("Loading unigrams...")
try:
    word_counts = Counter()
    with open('corpus.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().upper().split()
            word_counts.update(words)
    unigrams = list(word_counts.keys())
    print(f"Loaded {len(unigrams)} unigrams from corpus.txt")
except FileNotFoundError:
    print("Warning: corpus.txt not found. Using None for unigrams.")
    unigrams = None

# --- Load Test Dataset ---
print(f"Loading test data from '{TEST_CSV_PATH}'...")
test_dataset = LibriSpeechDataset(
    csv_path=TEST_CSV_PATH,
    labels=labels
)

if USE_SUBSET:
    # Use only a subset for faster optimization
    subset_indices = list(range(min(SUBSET_SIZE, len(test_dataset))))
    test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
    print(f"Using subset of {len(test_dataset)} samples for optimization")

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# --- Pre-compute Model Outputs ---
print("Pre-computing model outputs to speed up optimization...")
model_outputs = []
ground_truths = []

with torch.no_grad():
    for mfccs, transcripts, mfccs_lengths, _ in tqdm(test_loader, desc="Computing model outputs"):
        mfccs = mfccs.to(device)
        log_probs = model(mfccs)
        log_probs_numpy = log_probs.cpu().numpy()
        
        for j in range(log_probs_numpy.shape[0]):
            single_log_probs = log_probs_numpy[j][:mfccs_lengths[j]]
            model_outputs.append(single_log_probs)
            
            ground_truth_tensor = transcripts[j]
            ground_truth = "".join([labels[i] for i in ground_truth_tensor.tolist() if labels[i] != "_"])
            ground_truths.append(ground_truth)

print(f"Pre-computed outputs for {len(model_outputs)} samples")

# --- Optimization Function ---
def evaluate_alpha_beta(alpha, beta):
    """Evaluate WER for given alpha and beta values"""
    try:
        # Build decoder with current parameters
        decoder = build_ctcdecoder(
            labels=[l for l in labels if l != '_'],
            kenlm_model_path='lm.bin',
            unigrams=unigrams,
            alpha=alpha,
            beta=beta
        )
        
        predictions = []
        
        # Decode all pre-computed outputs
        for log_probs in model_outputs:
            beam_results = decoder.decode_beams(log_probs, beam_width=BEAM_WIDTH)
            
            if beam_results:
                best_beam = beam_results[0]
                decoded_transcription, _, _, _, _ = best_beam
            else:
                decoded_transcription = ""
                
            predictions.append(decoded_transcription)
        
        # Calculate WER
        wer = jiwer.wer(ground_truths, predictions)
        return wer
        
    except Exception as e:
        print(f"Error with alpha={alpha}, beta={beta}: {e}")
        return float('inf')

# --- Grid Search ---
print("\n--- Starting Grid Search for Optimal Alpha and Beta ---")
print(f"Testing {len(ALPHA_RANGE)} alpha values × {len(BETA_RANGE)} beta values = {len(ALPHA_RANGE) * len(BETA_RANGE)} combinations")

results = []
best_wer = float('inf')
best_params = None

# Create all combinations
param_combinations = list(itertools.product(ALPHA_RANGE, BETA_RANGE))

for i, (alpha, beta) in enumerate(tqdm(param_combinations, desc="Optimizing parameters")):
    start_time = time.time()
    wer = evaluate_alpha_beta(alpha, beta)
    end_time = time.time()
    
    results.append({
        'alpha': alpha,
        'beta': beta,
        'wer': wer,
        'time': end_time - start_time
    })
    
    if wer < best_wer:
        best_wer = wer
        best_params = (alpha, beta)
        print(f"\nNew best WER: {wer:.4f} (α={alpha}, β={beta})")
    
    # Print progress every 10 combinations
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{len(param_combinations)}, Current best WER: {best_wer:.4f}")

# --- Results Analysis ---
print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)

print(f"Best parameters found:")
print(f"  Alpha (language model weight): {best_params[0]}")
print(f"  Beta (word insertion bonus): {best_params[1]}")
print(f"  Best WER: {best_wer:.4f} ({best_wer*100:.2f}%)")

# Sort results by WER
results.sort(key=lambda x: x['wer'])

print(f"\nTop 10 parameter combinations:")
print("-" * 50)
for i, result in enumerate(results[:10]):
    print(f"{i+1:2d}. α={result['alpha']:4.1f}, β={result['beta']:4.1f} → WER: {result['wer']:.4f} ({result['wer']*100:.2f}%)")

# Save detailed results
with open('alpha_beta_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to 'alpha_beta_optimization_results.json'")

# --- Generate Heatmap Data ---
print(f"\nGenerating heatmap data...")
wer_matrix = np.zeros((len(BETA_RANGE), len(ALPHA_RANGE)))

for result in results:
    alpha_idx = ALPHA_RANGE.index(result['alpha'])
    beta_idx = BETA_RANGE.index(result['beta'])
    wer_matrix[beta_idx, alpha_idx] = result['wer']

# Save heatmap data
np.savetxt('wer_heatmap.csv', wer_matrix, delimiter=',', 
           header=','.join([f'alpha_{a}' for a in ALPHA_RANGE]))

print(f"Heatmap data saved to 'wer_heatmap.csv'")
print(f"Rows represent beta values: {BETA_RANGE}")
print(f"Columns represent alpha values: {ALPHA_RANGE}")

# --- Final Recommendations ---
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if best_wer < 0.34:  # If we improved from your baseline of 34.03%
    improvement = (0.3403 - best_wer) / 0.3403 * 100
    print(f"🎉 Improvement achieved! WER reduced by {improvement:.1f}%")
else:
    print("📊 Consider the following approaches:")

print(f"\n1. Use the optimal parameters: α={best_params[0]}, β={best_params[1]}")
print(f"2. If results are similar, prefer lower α values for faster decoding")
print(f"3. Consider testing with a larger beam width (current: {BEAM_WIDTH})")

# Analyze patterns
high_alpha_results = [r for r in results if r['alpha'] >= 1.0]
low_alpha_results = [r for r in results if r['alpha'] < 1.0]

if high_alpha_results and low_alpha_results:
    avg_high_alpha = np.mean([r['wer'] for r in high_alpha_results])
    avg_low_alpha = np.mean([r['wer'] for r in low_alpha_results])
    
    if avg_low_alpha < avg_high_alpha:
        print(f"4. Lower alpha values generally perform better (avg WER: {avg_low_alpha:.4f} vs {avg_high_alpha:.4f})")
    else:
        print(f"4. Higher alpha values generally perform better (avg WER: {avg_high_alpha:.4f} vs {avg_low_alpha:.4f})")

print(f"\n5. Test the optimized parameters on your full validation set to confirm results")
print(f"6. Consider fine-tuning your language model if WER is still high")

print("\n" + "="*60)
print(f"Optimization completed in {len(param_combinations)} evaluations")
print("="*60)