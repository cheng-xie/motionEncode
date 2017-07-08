import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

class AutoEncoder:
    def __init__(self, encoder, decoder, data, use_cuda=True):
        self.enc = encoder
        self.dec = decoder
        self.data = data
        self.use_cuda = use_cuda

        # Dimensions of samples
        self.sample_size = iter(data).next()[0].size()
        # Dimensions of encoding
        self.code_size = decoder.input_size()

        # Use MSELoss for now
        self.criterion = nn.MSELoss()

        if self.use_cuda:
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.criterion = self.criterion.cuda()

        self.enc_optim = optim.Adam(self.enc.parameters())
        self.dec_optim = optim.Adam(self.dec.parameters())

        self.data_iter = iter(data)

    def train(self, iters, batch_size):
        ''' Trains the Auto Encoder for some number of steps.
        '''
        tot_loss = 0
        for _ in range(iters):
            # Zero out the gradients
            self.enc.zero_grad()
            self.dec.zero_grad()

            # Sample data
            data_samples = Variable(self._sample_data(batch_size))
            if self.use_cuda:
                data_samples = data_samples.cuda()

            # Target is the data itself
            target = data_samples
            if self.use_cuda:
                target = target.cuda()

            # Encode the sample
            encoding = self.enc.forward(data_samples)
            # Decode the sample
            recon = self.dec.forward(encoding)
            # Calculate the loss
            loss = self.criterion(recon, target)
            # Calculate some gradients
            loss.backward()
            # Run update step
            self.enc_optim.step()
            self.dec_optim.step()

            tot_loss += loss.data[0]
            # print('Loss', loss.data[0])
        print('Loss', tot_loss/iters)

    def reconstruct(self, samples):
        if self.use_cuda:
            samples = samples.cuda()
        return self.dec.forward(self.enc.forward(samples))

    def encode(self, samples):
        if self.use_cuda:
            samples = samples.cuda()
        return self.enc.forward(samples)

    def decode(self, codes):
        if self.use_cuda:
            codes = codes.cuda()
        return self.dec.forward(codes)

    def _sample_data(self, num_samples):
        ''' Draws num_samples samples from the data
        '''
        try:
            return self.data_iter.next()
        except:
            self.data_iter = iter(self.data)
            return self.data_iter.next()

