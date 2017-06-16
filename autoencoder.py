import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

class AutoEncoder:
    def __init__(self, encoder, decoder, data):
        self.enc = encoder
        self.dec = decoder
        self.data = data

        # Dimensions of samples
        self.sample_size = iter(data).next()[0].size()
        # Dimensions of encoding
        self.code_size = decoder.input_size()

        self.criterion = nn.MSELoss()

        self.enc_optim = optim.Adam(self.enc.parameters())
        self.dec_optim = optim.Adam(self.dec.parameters())

        self.data_iter = iter(data)

    def train(self, iters, batch_size):
        ''' Trains the Auto Encoder for some number of steps
        '''
        for _ in range(iters):
            # Zero out the gradients
            self.enc.zero_grad()
            self.dec.zero_grad()

            # Sample data
            data_samples = Variable(self._sample_data(batch_size))
            # Target is the data itself
            target = data_samples
            # Encode the sample
            encoding = self.enc.forward(data_samples)
            # Decode the sample
            recon = self.dec.forward(encoding)
            # Calculate the loss
            loss = self.criterion(recon, target)
            # Calculate some gradients
            loss.backward()
            self.gen_optim.step()

            print('Loss', loss.data[0])

    def _sample_data(self, num_samples):
        ''' Draws num_samples samples from the data
        '''
        try:
            return self.data_iter.next()
        except:
            self.data_iter = iter(self.data)
            return self.data_iter.next()

