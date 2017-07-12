import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

class AutoEncoder:
    def __init__(self, encoder, decoder, use_cuda=True):
        self.enc = encoder
        self.dec = decoder
        self.use_cuda = use_cuda

        # Dimensions of samples
        # self.sample_size = self.enc.input_size()
        # Dimensions of encoding
        # self.code_size = self.dec.input_size()

        # Use MSELoss for now
        self.criterion = nn.MSELoss()

        if self.use_cuda:
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.criterion = self.criterion.cuda()

        self.enc_optim = optim.Adam(self.enc.parameters())
        self.dec_optim = optim.Adam(self.dec.parameters())

    def train(self, data, epochs, batch_size, iters_per_log = 1):
        '''
            Trains the Auto Encoder for some number of epochs.
        '''
        self.data = DataLoader(torch.FloatTensor(data), batch_size=batch_size, shuffle=True)
        tot_loss = 0
        log_iters_count = 0
        for epoch in range(epochs):
            for cur_iter, data_batch in enumerate(self.data):
                # Zero out the gradients
                self.enc.zero_grad()
                self.dec.zero_grad()

                # Sample data
                data_samples = Variable(data_batch)
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
                log_iters_count += 1

                if log_iters_count % iters_per_log == 0:
                    print('Epoch {} Iter {}'.format(epoch, cur_iter))
                    print('Loss', tot_loss/iters_per_log)
                    tot_loss = 0

    def reconstruct(self, samples):
        """
            Reconstruct some samples.
            Args:
                @samples    Should be a numpy array of size (samples, sample_size).
        """
        return self._reconstruct(Variable(torch.FloatTensor(samples)))

    def _reconstruct(self, samples):
        """
            Reconstruct some samples.
            Args:
                @samples    Should be a torch Variable of size (samples, sample_size).
        """
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

    def save_state_dict(self, path):
        param_dict = {'enc': self.enc.state_dict(), 'dec': self.dec.state_dict()}
        torch.save(param_dict, path)

    def load_state_dict(self, path):
        param_dict = torch.load(path)
        self.enc.load_state_dict(param_dict['enc'])
        self.dec.load_state_dict(param_dict['dec'])

