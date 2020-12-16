import torch

class AR(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, GPU, input_size, num_layers, kernel_size=(3, 3),
                 batch_first=True, bias=True, return_all_layers=False):
        super(AR, self).__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim[0])

    def forward(self, x):
        out = self.linear(x)
        return out