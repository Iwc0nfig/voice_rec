#This is the script that you have to run in order to create the corpus.txt
#Next you must create the lm . I use kenlm 


"""
lmplz -o 4 --text corpus.txt --arpa lm.arpa
build_binary lm.arpa lm.binary
"""

import os

input_dir = "LibriSpeech"
output_file = "corpus.txt"

with open(output_file, 'w',encoding='utf-8') as f:
  for root,dirs,files in os.walk(input_dir):
    for file in files:
      if file.endswith(".trans.txt"):
        with open(os.path.join(root,file), 'r', encoding='utf-8') as t:
          for line in t:
            parts = line.strip().split(" ",1)
            if len(parts) ==2:
              transcription = parts[1].strip()
              f.write(transcription + "\n")