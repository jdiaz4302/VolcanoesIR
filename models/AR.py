import torch
import torch.nn as nn

class AR(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, GPU, input_size, num_layers, kernel_size=(3, 3),
                 batch_first=True, bias=False, return_all_layers=False):
        super(AR, self).__init__()
        # Hard-coded for 'all' only
        print("WARNING: this model assumes training_data_set = 'all' by using 6 previous scenes")
        self.AR_coefs = nn.Parameter(torch.randn(6))

    def forward(self, x):
        out = torch.zeros(x.shape)
        for p in range(len(self.AR_coefs)):
            out[:, p, :] = self.AR_coefs[p]*x[:, p, :]
        out = torch.cumsum(out, dim = 1)
        return out