import json
from json import encoder
import os
import strider

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from models.mlpcoder import MLPEncoder, MLPDecoder
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def test_bimodal_motion(m_file_dir1, m_file_dir2, out_path):
    # preprocessing
    # TODO: handle different target frame rates
    dataset = None
    mdata = None
    window_size = 20
    stride = 5
    for m_file_path in os.listdir(m_file_dir1):
        with open(os.path.join(m_file_dir1, m_file_path)) as mfile:
            mdata = json.load(mfile)
            motion = np.array(mdata['Frames'], dtype=np.float32)
            samples = strider.stride_windows(motion, window_size=window_size, stride=stride)
            if dataset is None:
                dataset = samples
            else:
                dataset = np.concatenate((dataset, samples))

    if dataset is None:
        print('First dir did not contain proper data.')

    for m_file_path in os.listdir(m_file_dir2):
        with open(os.path.join(m_file_dir2, m_file_path)) as mfile:
            mdata = json.load(mfile)
            motion = np.array(mdata['Frames'], dtype=np.float32)
            samples = strider.stride_windows(motion, window_size=window_size, stride=stride)
            dataset = np.concatenate((dataset, samples))



    print(dataset.shape)
    # Flatten features as we are just using a MLP
    dataset = dataset.reshape(dataset.shape[0],-1)
    print(dataset.shape)

    num_iters = 250
    code_size = 3
    sample_size = dataset.shape[1]
    batch_size = dataset.shape[0]

    # leave index 0 as a validation sample
    data_loader = DataLoader(torch.FloatTensor(dataset[1:]), batch_size=batch_size, shuffle=True)

    enc = MLPEncoder(sample_size, code_size)
    dec = MLPDecoder(code_size, sample_size)
    aut = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    for ii in range(num_iters):
        print(str(ii) + '/' + str(num_iters) + ' ')
        aut.train(20, batch_size)

    # Try to reconstruct one of the samples
    recon = aut.reconstruct(Variable(torch.FloatTensor(dataset[0:1]))).data.cpu().numpy()
    recon = recon.reshape(recon.shape[0], window_size, -1)
    # Remove unnecessary dimension
    recon = recon.reshape(window_size, -1)
    print(recon)
    print(recon.shape)

    # postprocessing

    # visualizations
    codes = aut.encode(Variable(torch.FloatTensor(dataset))).data.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(codes[:,0],codes[:,1],codes[:,2],c=np.arange(codes.shape[0]))
    plt.show()

    # print to file
    mdata['Frames'] = recon.tolist()
    mdata['Loop'] = False
    with open(out_path, 'w') as outfile:
        encoder.FLOAT_REPR = lambda o: format(o, '.6f')
        json.dump(mdata, outfile)


def test_overfit_motion(m_file_dir, out_path):
    # preprocessing
    # TODO: handle different target frame rates
    dataset = None
    mdata = None
    window_size = 15
    stride = 3
    for m_file_path in os.listdir(m_file_dir):
        with open(os.path.join(m_file_dir, m_file_path)) as mfile:
            mdata = json.load(mfile)
            motion = np.array(mdata['Frames'], dtype=np.float32)
            samples = strider.stride_windows(motion, window_size=window_size, stride=3)
            if dataset is None:
                dataset = samples
            else:
                dataset = np.concatenate((dataset, samples))

    print(dataset.shape)
    # Flatten features as we are just using a MLP
    dataset = dataset.reshape(dataset.shape[0],-1)
    print(dataset.shape)

    num_iters = 250
    code_size = 3
    sample_size = dataset.shape[1]
    batch_size = dataset.shape[0]

    # leave index 0 as a validation sample
    data_loader = DataLoader(torch.FloatTensor(dataset[1:]), batch_size=batch_size, shuffle=True)

    enc = MLPEncoder(sample_size, code_size)
    dec = MLPDecoder(code_size, sample_size)
    aut = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    for ii in range(num_iters):
        print(str(ii) + '/' + str(num_iters) + ' ')
        aut.train(20, batch_size)

    # Try to reconstruct one of the samples
    recon = aut.reconstruct(Variable(torch.FloatTensor(dataset[0:1]))).data.cpu().numpy()
    recon = recon.reshape(recon.shape[0], window_size, -1)
    # Remove unnecessary dimension
    recon = recon.reshape(window_size, -1)
    print(recon)
    print(recon.shape)

    # postprocessing

    # visualizations
    codes = aut.encode(Variable(torch.FloatTensor(dataset))).data.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(codes[:,0],codes[:,1],codes[:,2],c=np.arange(codes.shape[0]))
    plt.show()

    # print to file
    mdata['Frames'] = recon.tolist()
    mdata['Loop'] = False
    with open(out_path, 'w') as outfile:
        encoder.FLOAT_REPR = lambda o: format(o, '.6f')
        json.dump(mdata, outfile)

