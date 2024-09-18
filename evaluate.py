# evaluate.py
import torch
from backend.phonetics import Alphabet
from backend.model.asr_model import ASRModel
from backend.data_mining.data_loader import get_dataloader

# Load the trained model
model = ASRModel(input_dim=13, hidden_dim=128, vocab_size=len(Alphabet))
model.load_state_dict(torch.load('G://Automatic Speecch Recognition System//model.pth'))

test_loader = get_dataloader('data/test-clean/mfcc', 'data/test-clean/transcript', batch_size=32)

# Evaluate
model.eval()
total_accuracy = 0
with torch.no_grad():
    for mfccs, transcripts in test_loader:
        logits = model(mfccs)
        predictions = logits.argmax(dim=-1)
        # Compute accuracy by comparing predictions to targets
        # (This is just an example, actual accuracy computation may vary based on your model's design)
