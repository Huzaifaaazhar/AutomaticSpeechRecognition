# utils/data_loading.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_mining.preprocessing import load_mfcc_files, load_transcripts, text_to_tokens, add_special_tokens

class ASRDataset(Dataset):
    def __init__(self, mfcc_dir, transcript_dir):
        self.mfccs = load_mfcc_files(mfcc_dir)
        self.transcripts = load_transcripts(transcript_dir)
    
    def __len__(self):
        return len(self.mfccs)
    
    def __getitem__(self, idx):
        mfcc = self.mfccs[idx]
        transcript = self.transcripts[idx]
        tokens = text_to_tokens(transcript)
        tokens = add_special_tokens(tokens)
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)

def get_dataloader(mfcc_dir, transcript_dir, batch_size=32):
    dataset = ASRDataset(mfcc_dir, transcript_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
