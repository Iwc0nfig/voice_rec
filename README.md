[README.md](https://github.com/user-attachments/files/21573877/README.md)
# voice_rec

## Overview

voice_rec is a Python-based speech recognition project that leverages a 1D CNN (flatten-linear) followed by an LSTM architecture for processing audio signals. The project utilizes PyTorch as the primary deep learning framework and incorporates several other key libraries for data processing and evaluation.

## Features

- Speech recognition using a custom neural network: 1D CNN → Flatten → Linear → LSTM.
- Built with PyTorch for flexible model development and training.
- Utilizes jiwer for word error rate evaluation, pandas and numpy for data manipulation, and librosa for audio processing.
- Language modeling with KenLM: create an ARPA format language model and convert it to binary for efficient decoding.

## Dependencies

- Python (version 3.7+ recommended)
- PyTorch
- jiwer
- pandas
- numpy
- librosa
- KenLM (for language modeling and ARPA to binary conversion)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Iwc0nfig/voice_rec.git
   cd voice_rec
   ```

2. Install Python dependencies:
   ```bash
   pip install torch jiwer pandas numpy librosa
   ```

3. Install KenLM for language modeling:
   - [KenLM installation guide](https://github.com/kpu/kenlm#installation)

## Usage

### Training the Model

1. Prepare your dataset according to the required format (see `data/` directory or scripts for details).
2. Train the model:
   ```bash
   python train.py --config config.yaml
   ```

### Running Inference

1. Use the trained model to transcribe audio:
   ```bash
   python infer.py --audio path/to/audio.wav --model path/to/model.pt
   ```

### Language Model with KenLM

1. Train a language model and generate an ARPA file:
   ```bash
   kenlm/bin/lmplz -o 3 < text_corpus.txt > lm.arpa
   ```
2. Convert the ARPA model to binary:
   ```bash
   kenlm/bin/build_binary lm.arpa lm.bin
   ```

## Evaluation

- Calculate Word Error Rate (WER) using jiwer:
  ```python
  from jiwer import wer
  ground_truth = "your reference text"
  hypothesis = "your model output"
  error = wer(ground_truth, hypothesis)
  print(f"WER: {error}")
  ```

## License

This project is provided under the terms of the repository's license.
