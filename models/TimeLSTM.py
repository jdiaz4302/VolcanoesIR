# Adapted from: https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb
import torch
import torch.nn as nn
from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

# Building the base TimeLSTM class
class TimeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_sz, GPU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_sz
        self.GPU = GPU
        # Factor of 5 (not 4) because input and forget are but there are two Ts
        self.weights_x = nn.Parameter(torch.randn(input_dim, hidden_sz * 5))
        # Factor of 3 because forget gate was lost
        self.weights_h = nn.Parameter(torch.randn(hidden_sz, hidden_sz * 3))
        # Additionally, time differences are used in T1...
        self.weights_t1 = nn.Parameter(torch.randn(1, hidden_sz))
        # And, separately (due to constraints): T2 and output
        self.weights_t = nn.Parameter(torch.randn(1, hidden_sz * 2))
        # Adapted for i, t1, t2, c, and o
        self.bias = nn.Parameter(torch.randn(hidden_sz * 5))
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, TimeDiff, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        if self.GPU:
            h_t, c_t = (h_t.cuda(), c_t.cuda())
        
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            TimeDiff_t = TimeDiff[:, t, :]
            # batch the computations into a single matrix multiplication
            # And apply all TimeLSTM equations
            # Input gate
            i_t = x_t @ self.weights_x[:, :HS] + h_t @ self.weights_h[:, :HS] + self.bias[:HS]
            i_t = torch.sigmoid(i_t)
            # Time one gate
            self.weights_t1 = torch.nn.Parameter(self.weights_t1.clamp(max = 0))
            t1_t_inner = TimeDiff_t @ self.weights_t1
            t1_t_inner = torch.tanh(t1_t_inner)
            t1_t = x_t @ self.weights_x[:, HS:HS*2] + t1_t_inner + self.bias[HS:HS*2]
            t1_t = torch.sigmoid(t1_t)
            # Time two gate
            t2_t = torch.tanh(TimeDiff_t @ self.weights_t[:, :HS])
            t2_t = x_t @ self.weights_x[:, HS*2:HS*3] + t2_t + self.bias[HS*2:HS*3]
            t2_t = torch.sigmoid(t2_t)
            # C shared components
            c_weight_application_t = x_t @ self.weights_x[:, HS*3:HS*4] + h_t @ self.weights_h[:, HS:HS*2]
            c_weight_application_t = torch.tanh(c_weight_application_t + self.bias[HS*3:HS*4])
            # Two C gates
            c_tilde_t = ((1 - (i_t * t1_t))*c_t) + (i_t * t1_t * c_weight_application_t)
            c_tilde_t = torch.sigmoid(c_tilde_t)
            c_t = ((1 - i_t)*c_t) + (i_t * t2_t * c_weight_application_t)
            c_t = torch.sigmoid(c_t)
            # Output
            o_t = x_t @ self.weights_x[:, HS*4:] + TimeDiff_t @ self.weights_t[:, HS:]
            o_t = o_t + h_t @ self.weights_h[:, HS*2:] + self.bias[HS*4:]
            o_t = torch.sigmoid(o_t)
            # Hidden
            h_t = o_t + torch.tanh(c_tilde_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


# Stacking the base TimeLSTM class 4 times for a deeper network
class StackedTimeLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, GPU, input_size=False, num_layers=False):
        """Simply stacking the simple TimeLSTM for multilayer model"""
        super(StackedTimeLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.GPU = GPU
        self.input_size = input_size # image h and w, relic from/for spatial models
        self.num_layers = num_layers # also relic
        # Wanting more/less than 4 layers will require manual editting
        assert(len(self.hidden_dim) == 4)
        self.layer1 = TimeLSTM(input_dim, self.hidden_dim[0], self.GPU)
        self.layer2 = TimeLSTM(self.hidden_dim[0], self.hidden_dim[1], self.GPU)
        self.layer3 = TimeLSTM(self.hidden_dim[1], self.hidden_dim[2], self.GPU)
        self.layer4 = TimeLSTM(self.hidden_dim[2], self.hidden_dim[3], self.GPU)

    def forward(self, x, t):
        h1, _ = self.layer1(x, t)
        h2, _ = self.layer2(h1, t)
        h3, _ = self.layer3(h2, t)
        o, _ = self.layer4(h3, t)
        return o, _