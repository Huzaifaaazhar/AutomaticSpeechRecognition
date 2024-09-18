# utils/preprocessing.py
import os
import numpy as np
from backend.phonetics import CMUdict_ARPAbet, Alphabet

# Load MFCC files
def load_mfcc_files(mfcc_dir):
    mfcc_files = [os.path.join(mfcc_dir, f) for f in os.listdir(mfcc_dir) if f.endswith('.npy')]
    return [np.load(file) for file in mfcc_files]

# Load transcript files (either raw or csv based on your structure)
def load_transcripts(transcript_dir, csv=False):
    if csv:
        # If transcripts are in CSV, handle accordingly
        transcripts = []
        with open(os.path.join(transcript_dir, 'random_submission.csv'), 'r') as f:
            next(f)  # Skip header
            for line in f:
                transcripts.append(line.strip().split(',')[1])  # Assuming second column contains text
        return transcripts
    else:
        transcript_files = [os.path.join(transcript_dir, 'raw', f) for f in os.listdir(os.path.join(transcript_dir, 'raw')) if f.endswith('.npy')]
        return [np.load(file).tolist() for file in transcript_files]

# Phoneme to ARPAbet conversion
def phonemes_to_arpabet(phoneme_sequence):
    return [CMUdict_ARPAbet.get(phoneme, '[UNK]') for phoneme in phoneme_sequence]

# Text to token conversion (alphabet indexing)
def text_to_tokens(text):
    return [Alphabet.index(char) if char in Alphabet else Alphabet.index('[UNK]') for char in text]

# Add special tokens for sequence modeling
def add_special_tokens(tokens):
    return [Alphabet.index('[SOS]')] + tokens + [Alphabet.index('[EOS]')]
