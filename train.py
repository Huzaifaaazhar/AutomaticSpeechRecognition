# train.py
import torch
import torch.optim as optim
from backend.phonetics import Alphabet
from backend.model.asr_model import ASRModel, ctc_loss
from backend.data_mining.data_loader import get_dataloader

# Hyperparameters
input_dim = 13  # MFCC feature dimension
hidden_dim = 128
vocab_size = len(Alphabet)
num_epochs = 10
learning_rate = 0.001

# Model, optimizer, and data loader
model = ASRModel(input_dim, hidden_dim, vocab_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = get_dataloader('data/train-clean-360/mfcc', 'data/train-clean-360/transcript', batch_size=32)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for mfccs, transcripts in train_loader:
        optimizer.zero_grad()
        
        input_lengths = torch.full(size=(mfccs.size(0),), fill_value=mfccs.size(1), dtype=torch.long)
        target_lengths = torch.full(size=(transcripts.size(0),), fill_value=transcripts.size(1), dtype=torch.long)
        
        logits = model(mfccs)
        
        loss = ctc_loss(logits, transcripts, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
