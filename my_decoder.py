import numpy as np
from pyctcdecode import build_ctcdecoder
from collections import Counter

logtis = "hellow men"

labels = [
    " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","'","_"
]


word_counts = Counter()
with open('corpus.txt', 'r', encoding='utf-8') as f:
    for line in f:
            # Simple tokenization: lowercase and split by spaces.
            # Adjust this if your corpus has different preprocessing.
        words = line.strip().split()
        word_counts.update(words)
            
    # Get a list of unique words
    unigrams = list(word_counts.keys())
print(len(unigrams))

kenlm_model_path = 'test.bin'  # <--- CHANGE THIS to the path of your .bin file
alpha = 0.6  # Language model weight
beta = 1.2   # Word insertion bonus

# --- 3. Build the Decoder ---
print("Loading the decoder...")
try:
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=kenlm_model_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta
    )
    print("Decoder loaded successfully.")
except Exception as e:
    print(f"Error loading decoder: {e}")
    # Exit if the decoder can't be loaded (e.g., file not found)
    exit()


# --- 4. Get Logits (Simulated) ---
# In your actual code, this will come from: `logits = your_model(audio_input)`
print("Generating dummy logits for demonstration...")
num_classes = len(labels)
# Shape: (batch_size, num_timesteps, num_classes)
logits = np.random.randn(1, 50, num_classes).astype(np.float32)
print(f"Logits shape: {logits.shape}")


# --- 5. Perform Decoding ---
print("\nDecoding with beam search...")
# Note: We pass logits[0] because the decoder works on a single audio sample, not a batch.
beam_results = decoder.decode_beams(logits=logits[0], beam_width=100)

# The first result in the list is the best one
decoder.decode_beams(logits=logits[0], beam_width=100)

# The first result is the most likely one
best_beam = beam_results[0]
print(best_beam)

# --- FIX IS HERE (using unpacking) ---
transcription, _, word_timestamps, log_prob, _ = best_beam
# ----------------------------------------

print(f"Decoded Transcription: {transcription}")
print(f"Confidence Score (Log Probability): {log_prob:.4f}")

# You can do this in a loop to see other candidates:
print("\nTop 5 candidate transcriptions:")
if beam_results:
    best_beam = beam_results[0]
    
    # FIX: Unpack the 5-element tuple correctly
    transcription, _, word_timestamps, log_prob, _ = best_beam
    
    print(f"\n--- DECODING RESULTS ---")
    print(f"Best Transcription: '{transcription}'")
    
    # The score is a log probability (negative). np.exp() converts it to a regular probability (0 to 1)
    confidence = np.exp(log_prob)
    print(f"Confidence Score: {confidence:.2%}")
    
    # You can also print the word timings!
    print("Word Timestamps:")
    for word, (start, end) in word_timestamps:
        print(f"- {word}: (from timestep {start} to {end})")
    print("--------------------------")

else:
    print("Decoding returned no results.")

