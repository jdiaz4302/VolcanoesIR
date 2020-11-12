# Copied from: https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb
import torch
import torch.nn as nn
from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

# Simple LSTM made from scratch
class LSTM(nn.Module):
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
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)
    
# Stacking the base LSTM class for a deeper network
class StackedLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, GPU, input_size=False, num_layers=False):
        """Simply stacking the simple TimeLSTM for multilayer model"""
        super(StackedLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.GPU = GPU
        self.input_size = input_size # image h and w, relic from/for spatial models
        self.num_layers = num_layers # also relic
        # Wanting more/less than 4 layers will require manual editting
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(LSTM(input_dim=cur_input_dim,
                                  hidden_dim=self.hidden_dim[i],
                                  GPU=self.GPU))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x):

        layer_output_list = []
        last_state_list   = []

        seq_len = x.size(1)

        for layer_idx in range(self.num_layers):
            output_inner = []
            for k in range(seq_len):
                if k == 0 and layer_idx == 0:
                    h, c = self.cell_list[layer_idx](x[:, [k], :])
                elif k == 0 and layer_idx != 0:
                    h, c = self.cell_list[layer_idx](x[:, k, :])
                # If not the first ele. in seq., use hidden state
                elif k != 0 and layer_idx != 0:
                    h, c = self.cell_list[layer_idx](x[:, k, :],
                                                     init_states=[c[0],c[1]])
                elif k != 0 and layer_idx == 0:
                    h, c = self.cell_list[layer_idx](x[:, [k], :],
                                                     init_states=[c[0],c[1]])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            layer_output.shape
            x = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        layer_output_list = layer_output_list[-1:]
        last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list