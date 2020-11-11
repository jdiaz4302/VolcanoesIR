# Adapted from: https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb
import torch
import torch.nn as nn
from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

# Building the base TimeAwareLSTM class
class TimeAwareLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, GPU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.GPU = GPU
        self.weights_x = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weights_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        # Additionally, weights for adjusting memory by time differences
        self.weights_t = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        # Adapted for i, f, o, c and c-new/adjusted
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 5))
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
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        if self.GPU:
            h_t, c_t = (h_t.cuda(), c_t.cuda())
        
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            TimeDiff_t = TimeDiff[:, t, :]
            # batch the computations into a single matrix multiplication
            # And apply all TimeAwareLSTM equations
            # Adjusting previous memory
            # Short term memory
            c_s_t_minus_1 = torch.tanh(c_t @ self.weights_t + self.bias[:HS])
            # Discounted short term memory, scalar assumption for TimeDiff
            c_hat_s_t_minus_1 = c_s_t_minus_1 * (-1 * torch.tanh(TimeDiff_t))
            # Long-term memory
            c_t_t_minus_1 = c_t - c_s_t_minus_1
            # Adjusted previous memory
            c_star_t_minus_1 = c_t_t_minus_1 + c_hat_s_t_minus_1
            # Input gate
            i_t = x_t @ self.weights_x[:, :HS] + h_t @ self.weights_h[:, :HS] + self.bias[HS:HS*2]
            i_t = torch.sigmoid(i_t)
            # Forget gate
            f_t = x_t @ self.weights_x[:, HS:HS*2] + h_t @ self.weights_h[:, HS:HS*2] + self.bias[HS*2:HS*3]
            f_t = torch.sigmoid(f_t)
            # Output gate
            o_t = x_t @ self.weights_x[:, HS*2:HS*3] + h_t @ self.weights_h[:, HS*2:HS*3] + self.bias[HS*3:HS*4]
            o_t = torch.sigmoid(o_t)
            # Candidate memory
            c_tilde = x_t @ self.weights_x[:, HS*3:] + h_t @ self.weights_h[:, HS*3:] + self.bias[HS*4:]
            c_tilde = torch.tanh(c_tilde)
            # Current memory
            c_t = (f_t * c_star_t_minus_1) + (i_t * c_tilde)
            # Current hidden state
            h_t = o_t + torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


# Stacking the base TimeAwareLSTM class 4 times for a deeper network
class StackedTimeAwareLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, GPU, input_size=False, num_layers=False):
        """Simply stacking the simple TimeLSTM for multilayer model"""
        super(StackedTimeAwareLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.GPU = GPU
        self.input_size = input_size # image h and w, relic from/for spatial models
        self.num_layers = num_layers # also relic
        # Wanting more/less than 4 layers will require manual editting
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(TimeAwareLSTM(input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dim[i],
                                           GPU=self.GPU))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, t):

        layer_output_list = []
        last_state_list   = []

        seq_len = x.size(1)

        for layer_idx in range(self.num_layers):
            output_inner = []
            for k in range(seq_len):
                if k == 0 and layer_idx == 0:
                    h, c = self.cell_list[layer_idx](x[:, [k], :],
                                                     t[:, [k], :])
                elif k == 0 and layer_idx != 0:
                    h, c = self.cell_list[layer_idx](x[:, k, :],
                                                     t[:, [k], :])
                # If not the first ele. in seq., use hidden state
                elif k != 0 and layer_idx != 0:
                    h, c = self.cell_list[layer_idx](x[:, k, :],
                                                     t[:, [k], :],
                                                     init_states=[c[0],c[1]])
                elif k != 0 and layer_idx == 0:
                    h, c = self.cell_list[layer_idx](x[:, [k], :],
                                                     t[:, [k], :],
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