# Copied from: https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb
import torch
import torch.nn as nn
from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

# Simple LSTM made from scratch
class Identity(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, GPU, input_size=False, num_layers=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.GPU = GPU
        self.input_size = input_size # image h and w, relic from/for spatial models
        self.num_layers = num_layers # also relic
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, init_states = None):
        return x
    
# Stacking the base LSTM class for a deeper network
class Identity2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, GPU, input_size=False, num_layers=False):
        """Simply stacking the simple TimeLSTM for multilayer model"""
        super(StackedLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.GPU = GPU
        self.input_size = input_size # image h and w, relic from/for spatial models
        self.num_layers = num_layers # also relic

    def forward(self, x):

        return x