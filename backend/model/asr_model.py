import torch
import numpy as np
from torch import nn

class ASRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(ASRModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

    def predict(self, mfcc_features):
        self.eval()
        with torch.no_grad():
            output = self.forward(mfcc_features)
        return output

# Load the model (make sure to adjust the input_dim and vocab_size as per your phonetics setup)
def load_asr_model():
    model = ASRModel(input_dim=13, hidden_dim=128, vocab_size=35)
    model.load_state_dict(torch.load("G://Automatic Speecch Recognition System//model.pth", map_location=torch.device('cpu'), weights_only=True))  # Added map_location for CPU if not using GPU
    model.eval()  # Set the model to evaluation mode
    return model
