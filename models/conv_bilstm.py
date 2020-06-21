import torch
import torch.nn as nn


"""ref: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py"""

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.

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

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2, kernel_size // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        c_0 = torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        return h_0, c_0

class BiLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.

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

        super(BiLSTM, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.lstm_fw = ConvLSTMCell(input_size, input_dim, hidden_dim, kernel_size, bias=bias)

        self.lstm_bw = ConvLSTMCell(input_size, input_dim, hidden_dim, kernel_size, bias=bias)

    def forward(self, x, h0_fw, c0_fw, h0_bw, c0_bw):
        seq_len = x.shape[1]
        h_list = [None] * seq_len #[None] * 2 = [None, None]

        h_prev, c_prev = h0_bw, c0_bw
        for t in range(seq_len-1, -1, -1):
            xt = x[:, t, :, :, :]
        
            ht, ct = self.lstm_bw(xt, (h_prev, c_prev))
            h_list[t] = ht
            h_prev, c_prev = ht, ct

            #print(h_prev.shape)
        h_bwout, c_bwout = h_prev, c_prev
        h_prev, c_prev = h0_fw, c0_fw
        out_list = [None] * seq_len
        for t in range(seq_len):
            xt = x[:, t, :, :, :]
        
            h_prev += h_list[t] # or concate
        
            ht, ct = self.lstm_fw(xt, (h_prev, c_prev))
            h_prev, c_prev = ht, ct
            out_list[t] = h_prev
        #return ht, ct, h_bkout, c_bwout
        return out_list, (ht, ct, h_bwout, c_bwout)

if __name__ == "__main__":
    input_size = (16, 32)
    input_dim = 3
    hidden_dim = 256 #10
    kernel_size = (3, 3)
    model = BiLSTM(input_size, input_dim, hidden_dim, kernel_size, bias=False).cuda()

    batch_size = 4
    seq_len = 2
    x = torch.randn(batch_size, seq_len, input_dim, 16, 32).cuda()

    h0_fw, c0_fw = model.lstm_fw.init_hidden(batch_size)
    h0_bw, c0_bw = model.lstm_bw.init_hidden(batch_size)

    h0_fw, c0_fw = h0_fw.cuda(), c0_fw.cuda()
    h0_bw, c0_bw = h0_bw.cuda(), c0_bw.cuda() 

    out_list, (ht, ct, h_bwout, c_bwout) = model(x, h0_fw, c0_fw, h0_bw, c0_bw)
    print(len(out_list), ht.shape, ct.shape, h_bwout.shape, c_bwout.shape)
    
"""
    h_list = [None] * seq_len
    h_prev, c_prev = h0_bw, c0_bw
    for t in range(seq_len-1, -1, -1):
        xt = x[:, t, :, :, :]
        
        ht, ct = model.lstm_bw(xt, (h_prev, c_prev))
        h_list[t] = ht
        h_prev, c_prev = ht, ct
        #print(h_prev.shape)

    h_prev, c_prev = h0_fw, c0_fw
    for t in range(seq_len):
        xt = x[:, t, :, :, :]
        
        h_prev += h_list[t] # or concate
        
        ht, ct = model.lstm_fw(xt, (h_prev, c_prev))
        h_prev, c_prev = ht, ct
        #print(h_prev)
"""
