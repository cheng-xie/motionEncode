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

                self.dis.zero_grad()

                # Backpropogate on true data
                target = Variable(torch.ones(batch_size))
                prediction = self.dis.forward(data_samples)
                loss_real = self.criterion(prediction, target)
                loss_real.backward()

                # Backpropogate on fake data
                target = Variable(torch.zeros(batch_size))
                data_samples = self.sample_gen(batch_size)
                # Don't propogate through generator
                data_samples = data_samples.detach()
                prediction = self.dis.forward(data_samples)
                loss_fake = self.criterion(prediction, target)
                loss_fake.backward()

                self.dis_optim.step()
                tot_loss_fake += loss_fake.data[0]
                tot_loss_real += loss_real.data[0]

            print('Disc', tot_loss_fake, ' ', tot_loss_real)

            self.gen.zero_grad()

            # Sample data
            data_samples = Variable(self._sample_data(batch_size))
            # Target is the data itself
            target = data_samples
            #
            encoding = self.enc.forward(data_samples)
            loss = self.criterion(prediction, target)
            loss.backward()

            print('Gen', loss.data[0])

            self.gen_optim.step()

    def sample_gen(self, num_samples):
        ''' Draws num_samples samples from the generator
        '''
        # Generate random noise variable from Normal dist
        noise = torch.randn(num_samples, self.noise_size)
        noise = Variable(noise)
        fake_samples = self.gen.forward(noise)
        return fake_samples

    def _sample_data(self, num_samples):
        ''' Draws num_samples samples from the data
        '''
        try:
            return self.data_iter.next()
        except:
            self.data_iter = iter(self.data)
            return self.data_iter.next()

