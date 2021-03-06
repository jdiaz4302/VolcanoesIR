




import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data


class ConvTime_LSTM1Cell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, GPU):
        """
        Initialize ConvTime_LSTM1 cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvTime_LSTM1Cell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        self.GPU         = GPU
        
        ## Defining the input convolutional layer ##
        self.i_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        
        ## Defining the T2 convolutional layer ##
        self.T1_conv = nn.Conv2d(in_channels=self.input_dim,
                                 out_channels=self.hidden_dim,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 bias=self.bias)
        
        ## Defining the T1 convolutional layer ##
        self.T2_conv = nn.Conv2d(in_channels=self.input_dim,
                                 out_channels=self.hidden_dim,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 bias=self.bias)
        
        ## Defining the activation convolutional layer ##
        self.c_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        
        ## Defining the output convolutional layer ##
        self.o_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        
        ## Defining the learnable parameter for time scaling ##
        self.constrained_scalar = nn.Parameter(torch.rand(1, 1))
        self.unconstrained_scalars = nn.Parameter(torch.rand(2, 1))

    def forward(self, input_tensor, time_tensor, cur_state):
        
        
        ## Getting the h_{m-1} and c_{m-1} ##
        ##     the previous hidden and activations ##
        h_cur, c_cur = cur_state
        
        
        ## Getting the (time tensor)*(their learned scalar) ##
        ## For T1, with a constrained weight ##
        t_T1 = self.constrained_scalar.clamp(max = 0) * time_tensor
        ## For T2 and output gate, without constraints ##
        t_bands_list = []
        for i in range(2):
            unconstrained_scalar = self.unconstrained_scalars[i]
            scaled_time_tensors = time_tensor * unconstrained_scalar
            t_bands_list.append(scaled_time_tensors)
        t_T2, t_o = t_bands_list
        ## Applying the appropriate non-linearities to t_T1 and t_T2 ##
        t_T1 = torch.sigmoid(t_T1)
        t_T2 = torch.sigmoid(t_T2)


        ## concatenate the prev. hidden state and the current input along the color channel dim ##
        x_h_combined = torch.cat([input_tensor, h_cur], dim = 1)
        
        
        ## The input gate ##
        ## Running the input convolution ##
        i_conv_outputs = self.i_conv(x_h_combined)
        ## Running the input LSTM gate equations ##
        i_m = torch.sigmoid(i_conv_outputs)
        
        
        ## The first time gate ##
        ## Running the first time convolution ##
        T1_conv_outputs = self.T1_conv(input_tensor)
        print(T1_conv_outputs.shape)
        print(t_T1.shape)
        T1_conv_outputs = T1_conv_outputs + t_T1
        ## Running the T1 LSTM gate equations ##
        T1_m = torch.sigmoid(T1_conv_outputs)
        
        
        ## The second time gate ##
        ## Running the second time convolution ##
        T2_conv_outputs = self.T2_conv(input_tensor)
        T2_conv_outputs = T2_conv_outputs + t_T2
        ## Running the T2 LSTM gate equations ##
        T2_m = torch.sigmoid(T2_conv_outputs)
        
        
        ## The c vectors ##
        ## Running the c convolution ##
        c_conv_outputs = self.c_conv(x_h_combined)
        ## Computing the c tilde and c activation vectors ##
        c_m_tilde = (((1 - i_m * T1_m) * c_cur) +
                     (i_m * T1_m * torch.tanh(c_conv_outputs)))
        c_m = (((1 - i_m) * c_cur) +
               (i_m * T2_m * torch.tanh(c_conv_outputs)))
        
         
        ## The output gate ##
        ## Running the output gate convolution
        o_conv_output = self.o_conv(x_h_combined)
        o_conv_output = o_conv_output + t_o
        ## Running the output LSTM gate equations
        o_m = torch.sigmoid(o_conv_output)
        
        
        ## The hidden vector ##
        h_m = o_m * torch.tanh(c_m_tilde)
        
        
        return h_m, c_m

    def init_hidden(self, batch_size):
        to_return = (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                     Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))
        if self.GPU:
            to_return = (to_return[0].cuda(), to_return[1].cuda())
        return(to_return)


class ConvTime_LSTM1(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers, GPU):
        super(ConvTime_LSTM1, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.GPU = GPU

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvTime_LSTM1Cell(input_size=(self.height, self.width),
                                                input_dim=cur_input_dim,
                                                hidden_dim=self.hidden_dim[i],
                                                kernel_size=self.kernel_size[i],
                                                bias=self.bias,
                                                GPU=self.GPU))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, time_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        cur_time_input = time_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor = cur_layer_input[:, t, :, :, :],
                                                 time_tensor = cur_time_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param