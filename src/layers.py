import torch
from torch import nn
import math

# custom linear layer with low-rank representation of weight matrix
# started with custom layer from https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
class LoRALinear(nn.Module):
    def __init__(self, size_in, size_out, rank):
        super().__init__()
        self.size_in, self.size_out, self.rank = size_in, size_out, rank
        weights_A = torch.Tensor(size_out, rank)
        weights_B = torch.Tensor(rank,  size_in)
        self.weights_A = nn.Parameter(weights_A)
        self.weights_B = nn.Parameter(weights_B)
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.uniform_(self.weights_A, -1./math.sqrt(rank), 1./math.sqrt(rank)) # Is this initialization (provably) good ?!
        nn.init.uniform_(self.weights_B, -1./math.sqrt(rank), 1./math.sqrt(rank))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights_B)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        # NOTE: x * (A * B)^T = (x * B^T) * A^T, which is more efficient to compute
        # w_times_x= torch.mm(x, torch.mm(self.weights_A, self.weights_B).t()) # naive, slow implementation
        w_times_x = torch.mm(torch.mm(x, self.weights_B.t()), self.weights_A.t())
        return torch.add(w_times_x, self.bias)  # w times x + b