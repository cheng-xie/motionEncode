import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)

class MLPDecoder(nn.Module):
    def __init__(self, d_in, d_out):
        super(MLPDecoder, self).__init__()
        H1 = 10
        H2 = 100
        self._d_in = d_in
        self._d_out = d_out

        # noise -> hidden
        self.l1 = nn.Linear(d_in, H1)
        self.l11 = nn.Linear(H1, H2)
        # hidden -> sample
        self.l2 = nn.Linear(H2, d_out)

    def forward(self, input):
        x = self.l1(input)
        x = F.relu(x)
        x = self.l11(x)
        x = F.relu(x)
        x = self.l2(x)
        # x = F.tanh(x)
        return x

    def input_size(self):
        return self._d_in

class MLPEncoder(nn.Module):
    def __init__(self, d_in, d_out):
        super(MLPEncoder, self).__init__()
        H1 = 100
        H2 = 10
        self.d_in = d_in

        # sample -> hidden
        self.l1 = nn.Linear(d_in, H1)
        self.l11 = nn.Linear(H1, H2)
        # hidden -> probability
        self.l2 = nn.Linear(H2, d_out)

    def forward(self, input):
        x = self.l1(input)
        x = F.relu(x)
        x = self.l11(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.tanh(x)
        return x

    def input_size(self):
        return self._d_in

