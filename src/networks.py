import torch
from torch import nn

from layers import LoRALinear

# Define baseline model with full-rank weights
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits

# Define model with low-rank weights
class LoRANeuralNetwork(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            LoRALinear(28*28, 512, rank),
            nn.ReLU(),
            LoRALinear(512, 512, rank),
            nn.ReLU(),
            LoRALinear(512, 10, rank)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits

# Deine model with "autoencoder-layers", i.e. two neighboring layers with m -> rank -> n neurons
class AENeuralNetwork(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(28*28, rank, bias=False),
            nn.Linear(rank, 512),
            nn.ReLU(),
            nn.Linear(512, rank, bias=False),
            nn.Linear(rank, 512),
            nn.ReLU(),
            nn.Linear(512, rank, bias=False),
            nn.Linear(rank, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits