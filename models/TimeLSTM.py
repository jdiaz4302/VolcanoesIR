x_and_t_channels_fragile = 5

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
    def __init__(self, input_dim, hidden_dim, GPU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.GPU = GPU
        # Factor of 3 (not 4) because input and forget are coupled
        self.weights_x = nn.Parameter(torch.randn(input_dim, hidden_dim * 3))
        # Factor of 2 for the time-based calculations that don't change with width
        self.weights_x_maintained = nn.Parameter(torch.randn(x_and_t_channels_fragile, hidden_dim * 2))
        # Factor of 3 because forget gate was lost
        self.weights_h = nn.Parameter(torch.randn(hidden_dim, hidden_dim * 3))
        # Additionally, time differences are used in T1...
        self.weights_t1 = nn.Parameter(torch.randn(1, hidden_dim))
        # And, separately (due to constraints): T2 and output
        self.weights_t = nn.Parameter(torch.randn(1, hidden_dim * 2))
        # Adapted for i, t1, t2, c, and o
        self.bias = nn.Parameter(torch.randn(hidden_dim * 5))
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x_for_h, x_for_x, TimeDiff, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x_for_h.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x_for_h.device), 
                        torch.zeros(bs, self.hidden_size).to(x_for_h.device))
        else:
            h_t, c_t = init_states
        if self.GPU:
            h_t, c_t = (h_t.cuda(), c_t.cuda())
        
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x_for_h[:, t, :]
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
            t1_t = x_for_x[:, t, :] @ self.weights_x_maintained[:, :HS] + t1_t_inner + self.bias[HS:HS*2]
            t1_t = torch.sigmoid(t1_t)
            # Time two gate
            t2_t = torch.tanh(TimeDiff_t @ self.weights_t[:, :HS])
            t2_t = x_for_x[:, t, :] @ self.weights_x_maintained[:, HS:HS*2] + t2_t + self.bias[HS*2:HS*3]
            t2_t = torch.sigmoid(t2_t)
            # C shared components
            c_weight_application_t = x_t @ self.weights_x[:, HS:HS*2] + h_t @ self.weights_h[:, HS:HS*2]
            c_weight_application_t = torch.tanh(c_weight_application_t + self.bias[HS*3:HS*4])
            # Two C gates
            c_tilde_t = ((1 - (i_t * t1_t))*c_t) + (i_t * t1_t * c_weight_application_t)
            c_tilde_t = torch.sigmoid(c_tilde_t)
            c_t = ((1 - i_t)*c_t) + (i_t * t2_t * c_weight_application_t)
            c_t = torch.sigmoid(c_t)
            # Output
            o_t = x_t @ self.weights_x[:, HS*2:] + TimeDiff_t @ self.weights_t[:, HS:]
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
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(TimeLSTM(input_dim=cur_input_dim,
                                      hidden_dim=self.hidden_dim[i],
                                      GPU=self.GPU))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x_for_h, x_for_x, t):

        layer_output_list = []
        last_state_list   = []

        seq_len = x_for_h.size(1)

        for layer_idx in range(self.num_layers):
            output_inner = []
            for k in range(seq_len):
                if k == 0 and layer_idx == 0:
                    h, c = self.cell_list[layer_idx](x_for_h[:, [k], :],
                                                     x_for_x[:, [k], :],
                                                     t[:, [k], :])
                elif k == 0 and layer_idx != 0:
                    h, c = self.cell_list[layer_idx](x_for_h[:, k, :],
                                                     x_for_x[:, [k], :],
                                                     t[:, [k], :])
                # If not the first ele. in seq., use hidden state
                elif k != 0 and layer_idx != 0:
                    h, c = self.cell_list[layer_idx](x_for_h[:, k, :],
                                                     x_for_x[:, [k], :],
                                                     t[:, [k], :],
                                                     init_states=[c[0],c[1]])
                elif k != 0 and layer_idx == 0:
                    h, c = self.cell_list[layer_idx](x_for_h[:, [k], :],
                                                     x_for_x[:, [k], :],
                                                     t[:, [k], :],
                                                     init_states=[c[0],c[1]])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            layer_output.shape
            x_for_h = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        layer_output_list = layer_output_list[-1:]
        last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list