import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from backend.model.asr_model import ASRModel
from torch.utils.data import DataLoader, Dataset

# Define your dataset
class ASRDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)  # Convert to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert to tensor and ensure dtype is long

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Initialize the model, dataset, dataloader, optimizer, and loss function
model = ASRModel(input_dim=13, hidden_dim=128, vocab_size=35)

# Dummy dataset
features = np.random.randn(100, 10, 13).astype(np.float32)  # Example features (100 samples, 10 timesteps, 13 features)
labels = np.random.randint(0, 35, size=(100, 10))  # Example labels (100 samples, 10 timesteps)

dataset = ASRDataset(features=features, labels=labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):  # Example: 10 epochs
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, 35), targets.view(-1))  # Reshape for CrossEntropyLoss
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), 'model.pth')
