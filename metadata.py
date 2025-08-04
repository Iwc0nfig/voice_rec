import os
import csv

def create_librispeech_csv(dataset_path, output_csv_file):
    """
    Creates a CSV file containing file paths and transcriptions
    for the LibriSpeech dataset.

    Args:
        dataset_path (str): The path to the root of the LibriSpeech dataset
                            (e.g., './LibriSpeech/train-clean-100').
        output_csv_file (str): The path to save the output CSV file.
    """
    print(f"Searching for transcription files in: {dataset_path}")
    
    # Check if the dataset path exists
    if not os.path.isdir(dataset_path):
        print(f"Error: The directory '{dataset_path}' does not exist.")
        print("Please make sure you have extracted the dataset and provided the correct path.")
        return

    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header of the CSV file
        csv_writer.writerow(['filepath', 'text'])

        # Walk through the directory structure
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".trans.txt"):
                    transcription_file_path = os.path.join(root, file)
                    
                    with open(transcription_file_path, 'r') as f:
                        for line in f:
                            # Split the line into utterance ID and the transcription
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                utterance_id, text = parts
                                
                                # Construct the path to the corresponding FLAC audio file
                                audio_filename = f"{utterance_id}.flac"
                                audio_filepath = os.path.join(root, audio_filename)
                                
                                # Check if the audio file actually exists
                                if os.path.exists(audio_filepath):
                                    # Write the absolute path and the text to the CSV
                                    csv_writer.writerow([os.path.abspath(audio_filepath), text])
                                else:
                                    print(f"Warning: Audio file not found for utterance: {utterance_id}")

    print(f"Successfully created CSV file at: {output_csv_file}")

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#                 HOW TO USE THIS SCRIPT
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


# 2. Set the desired name and location for your output CSV file.
train_path = 'train-clean-100'
metadata = 'metadata.csv'

val_path = 'test-clean'
output_csv = 'val.csv'

# 3. Run the function.
create_librispeech_csv(train_path,metadata)
create_librispeech_csv(val_path, output_csv)