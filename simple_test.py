import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from models.mlpcoder import MLPEncoder, MLPDecoder
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import subprocess

def test_circ(mean, std, num_data, make_gif=False):
    '''
    Simple test, reconstruct points sampled uniformly from a fixed linesegment
    but only having one dimensional coding layer.
    '''
    num_iters = 3000
    code_size = 1
    sample_size = 2
    batch_size = 5000

    slope = 1
    offset = -0.1

    # Sample randomly from the linesegment
    def sample_lineseg(x0, x1, num, rand=True):
        if rand:
            x = np.random.uniform(x0, x1, size=num)
        else:
            x = np.linspace(x0, x1, num)

        y = x * slope + offset
        samples = (np.stack((x, y), axis=1))
        return samples

    # Sample randomly from circle
    def sample_circ(r, thet0, thet1, num, rand=True):
        if rand:
            thet = np.random.uniform(thet0, thet1, size=num)
        else:
            thet = np.linspace(thet0, thet1, num)
        x = r*np.cos(thet)
        y = r*np.sin(thet)
        samples = (np.stack((x, y), axis=1))
        return samples

    #data = torch.FloatTensor(sample_lineseg(0, 1, num_data))
    data = torch.FloatTensor(sample_circ(0.6, 0, 2*3.14, num_data))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # data_iter = iter(data_loader)
    # Construct an AutoEncoder
    enc = MLPEncoder(sample_size, code_size)
    dec = MLPDecoder(code_size, sample_size)
    aut = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    plt.ion()
    for ii in range(num_iters):
        aut.train(20, batch_size)
        # Sample and visualize
        #samples = sample_lineseg(-0.5, 1.5, 50, rand=False)
        samples = sample_circ(0.5,0.0, 2*3.14, 50, rand=False)

        # True Distribution
        plt.scatter(samples[:,0], samples[:,1], color="red", label="True")
        #plt.plot(bin_centers,y,'-')

        recon = aut.reconstruct(Variable(torch.FloatTensor(samples))).data.cpu().numpy()
        plt.scatter(recon[:,0], recon[:,1], color="blue", label="Recon")

        plt.legend()
        plt.xlim(-1.5, 1.5)
        # plt.ylim(-0.5, 1.5)
        if make_gif:
            plt.savefig('./figs/pic' + str(ii).zfill(3))

        plt.pause(0.01)
        plt.cla()

    if make_gif:
        subprocess.call([ 'convert', '-loop', '0', '-delay', '50', './figs/pic*.png', './figs/output.gif'])


def test_lineseg(mean, std, num_data, make_gif=False):
    '''
    Simple test, reconstruct points sampled uniformly from a fixed linesegment
    but only having one dimensional coding layer.
    '''
    num_iters = 3000
    code_size = 1
    sample_size = 2
    batch_size = 1000

    slope = 1
    offset = -0.1

    # Sample randomly from the linesegment
    def sample_lineseg(x0, x1, num, rand=True):
        if rand:
            x = np.random.uniform(x0, x1, size=num)
        else:
            x = np.linspace(x0, x1, num)

        y = x * slope + offset
        samples = (np.stack((x, y), axis=1))
        return samples

    # Sample randomly from circle
    def sample_circ(r, thet0, thet1, num, rand=True):
        if rand:
            thet = np.random.uniform(thet0, thet1, size=num)
        else:
            thet = np.linspace(thet0, thet1, num)
        x = r*np.cos(thet)
        y = r*np.sin(thet)
        samples = (np.stack((x, y), axis=1))
        return samples

    #data = torch.FloatTensor(sample_lineseg(0, 1, num_data))
    data = torch.FloatTensor(sample_circ(0.6, 0, 2*3.14, num_data))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # data_iter = iter(data_loader)
    # Construct an AutoEncoder
    enc = MLPEncoder(sample_size, code_size)
    dec = MLPDecoder(code_size, sample_size)
    aut = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    plt.ion()
    for ii in range(num_iters):
        aut.train(20, batch_size)
        # Sample and visualize
        #samples = sample_lineseg(-0.5, 1.5, 50, rand=False)
        samples = sample_circ(0.5,0.0, 2*3.14, 50, rand=False)

        # True Distribution
        plt.scatter(samples[:,0], samples[:,1], color="red", label="True")
        #plt.plot(bin_centers,y,'-')

        recon = aut.reconstruct(Variable(torch.FloatTensor(samples))).data.cpu().numpy()
        plt.scatter(recon[:,0], recon[:,1], color="blue", label="Recon")

        plt.axvline(x=0, ymin=0, ymax=1)
        plt.axvline(x=1, ymin=0, ymax=1)

        plt.legend()
        plt.xlim(-0.5, 1.5)
        # plt.ylim(-0.5, 1.5)
        if make_gif:
            plt.savefig('./figs/pic' + str(ii).zfill(3))

        plt.pause(0.01)
        plt.cla()

    if make_gif:
        subprocess.call([ 'convert', '-loop', '0', '-delay', '50', './figs/pic*.png', './figs/output.gif'])


