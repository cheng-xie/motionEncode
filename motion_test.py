import json
from json import encoder
import os

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from autoencoder.models.mlpcoder import MLPEncoder, MLPDecoder
from autoencoder.autoencoder import AutoEncoder
import util
from preprocess import TerrainRLMotionPreprocessor as Preprocessor

def test_bimodal_motion(mfile_dir1, mfile_dir2, out_path):
    # Preprocessing
    # TODO: handle different target frame rates
    dataset = None
    window_size = 40
    stride = 5
    do_preprocess = True
    do_visualize = True

    # Load data
    if do_preprocess:
        file_paths = util.file_list_from_dir_list([mfile_dir1, mfile_dir2])
        preprocessor = Preprocessor(t_feature_idx=0, x_feature_idx=1, z_feature_idx=3, window_size=window_size, stride=stride)
        dataset = preprocessor.generate_dataset(file_paths)
    else:
        dataset = util.add_dir_motion_windows(mfile_dir1, window_size = window_size, stride = stride, dataset = dataset)
        if dataset is None:
            print('First dir did not contain proper data.')
        dataset = util.add_dir_motion_windows(mfile_dir2, window_size = window_size, stride = stride, dataset = dataset)

    # Flatten features as we are just using a MLP
    dataset = dataset.reshape(dataset.shape[0],-1)


    # Train
    num_iters = 550
    num_sub_steps = 20
    code_size = 5
    sample_size = dataset.shape[1]
    # Full batch
    batch_size = dataset.shape[0]
    # Leave index 0 as a validation sample
    # data_loader = DataLoader(torch.FloatTensor(dataset[1:]), batch_size=batch_size, shuffle=True)
    data_loader = DataLoader(torch.FloatTensor(dataset), batch_size=batch_size, shuffle=True)

    enc = MLPEncoder(sample_size, code_size)
    dec = MLPDecoder(code_size, sample_size)
    aut = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    for ii in range(num_iters):
        print(str(ii) + '/' + str(num_iters) + ' ')
        aut.train(num_sub_steps, batch_size)


    # Visualizations
    if do_visualize and code_size >= 3:
        codes = aut.encode(Variable(torch.FloatTensor(dataset))).data.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(codes[:,0],codes[:,1],codes[:,2],c=np.arange(codes.shape[0]))
        plt.show()


    # Try to reconstruct one of the samples
    recon = aut.reconstruct(Variable(torch.FloatTensor(dataset[0:1]))).data.cpu().numpy()
    recon = recon.reshape(recon.shape[0], window_size, -1)
    # Remove unnecessary dimension
    recon = recon.reshape(window_size, -1)

    # postprocessing
    if do_preprocess:
        recon = preprocessor.convert_back(recon)

    print(recon)
    print(recon.shape)

    # write to file
    mdata = {}
    mdata['Frames'] = recon.tolist()
    mdata['Loop'] = False
    with open(out_path, 'w') as outfile:
        encoder.FLOAT_REPR = lambda f: ("%.2f" % f)
        json.dump(mdata, outfile)

def test_overfit_motion(m_file_dir, out_path):
    # preprocessing
    # TODO: handle different target frame rates
    dataset = None
    window_size = 15
    stride = 3

    dataset = util.add_dir_motion_windows(mfile_dir2, window_size = window_size, stride = stride, dataset = dataset)

    # Flatten features as we are just using a MLP
    dataset = dataset.reshape(dataset.shape[0],-1)

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
    mdata = {}
    mdata['Frames'] = recon.tolist()
    mdata['Loop'] = False
    with open(out_path, 'w') as outfile:
        encoder.FLOAT_REPR = lambda o: format(o, '.6f')
        json.dump(mdata, outfile)

