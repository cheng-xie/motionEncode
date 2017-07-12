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

class MultimodalMotionScenario:
    def __init__():
        init_model(sample_dims, code_dims)
        pass

    def init_model(sample_dims, code_dims):
        enc = MLPEncoder(sample_dims, code_dims)
        dec = MLPDecoder(code_dims, sample_dims)
        self.model = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    def save_param_dict(path):
        aut.save_state_dict(path)

    def load_param_dict(path):
        aut.load_state_dict(path)

    def train(dataset, num_iters, num_sub_steps, batch_size=-1):
        sample_dims = dataset.shape[1]
        if batch_size == -1:
            batch_size = dataset.shape[0]

        data_loader = DataLoader(torch.FloatTensor(dataset), batch_size=batch_size, shuffle=True)

        for ii in range(num_iters):
            print(str(ii) + '/' + str(num_iters) + ' ')
            aut.train(num_sub_steps, batch_size)

    def recon():
        """
            Reconstructs an input clip using our model.
        """
        # Try to reconstruct one of the samples
        recon = aut.reconstruct(Variable(torch.FloatTensor(dataset[100:101]))).data.cpu().numpy()
        recon = recon.reshape(recon.shape[0], window_size, -1)
        # Remove unnecessary dimension
        recon = recon.reshape(window_size, -1)

        # postprocessing
        if do_preprocess:
            recon = preprocessor.convert_back(recon)

        print(recon)
        print(recon.shape)

    def visualize():
        # Visualizations
        if do_visualize and code_dims >= 3:
            codes = aut.encode(Variable(torch.FloatTensor(dataset))).data.cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(codes[:,0],codes[:,1],codes[:,2],c=np.arange(codes.shape[0]))
            plt.show()

    def load_dataset():
        raise NotImplementedError("Subclasses should implement this!")

    def generate_dataset(mfile_dirs):
        # Preprocessing
        # TODO: handle different target frame rates
        dataset = None
        window_size = 30
        stride = 5
        do_preprocess = True
        do_visualize = True

        # Load data
        if do_preprocess:
            file_paths = util.file_list_from_dir_list(mfile_dirs)
            preprocessor = Preprocessor(t_feature_idx=0, x_feature_idx=1, z_feature_idx=3, window_size=window_size, stride=stride)
            dataset = preprocessor.generate_dataset(file_paths)
        else:
            for mfile_dir in mfile_dirs:
                dataset = util.add_dir_motion_windows(mfile_dir, window_size = window_size, stride = stride, dataset = dataset)
                if dataset is None:
                    print('Dir did not contain proper data.')

        # Flatten features as we are just using a MLP
        dataset = dataset.reshape(dataset.shape[0],-1)


def simple_test_multimodal_motion(mfile_dirs, out_path, save_weights_path, load_weights_path = None):
    dataset = None
    window_size = 30
    stride = 5
    do_preprocess = True
    do_visualize = True

def test_multimodal_motion(mfile_dirs, out_path, save_weights_path, load_weights_path = None):
    # Preprocessing
    # TODO: handle different target frame rates
    dataset = None
    window_size = 30
    stride = 5
    do_preprocess = True
    do_visualize = True

    # Load data
    if do_preprocess:
        file_paths = util.file_list_from_dir_list(mfile_dirs)
        preprocessor = Preprocessor(t_feature_idx=0, x_feature_idx=1, z_feature_idx=3, window_size=window_size, stride=stride)
        dataset = preprocessor.generate_dataset(file_paths)
    else:
        for mfile_dir in mfile_dirs:
            dataset = util.add_dir_motion_windows(mfile_dir, window_size = window_size, stride = stride, dataset = dataset)
            if dataset is None:
                print('Dir did not contain proper data.')

    # Flatten features as we are just using a MLP
    dataset = dataset.reshape(dataset.shape[0],-1)


    # Train
    num_iters = 1550
    num_sub_steps = 20
    code_size = 3
    sample_size = dataset.shape[1]
    # Full batch
    batch_size = dataset.shape[0]
    # Leave index 0 as a validation sample
    data_loader = DataLoader(torch.FloatTensor(dataset[1:]), batch_size=batch_size, shuffle=True)
    # data_loader = DataLoader(torch.FloatTensor(dataset), batch_size=batch_size, shuffle=True)

    enc = MLPEncoder(sample_size, code_size)
    dec = MLPDecoder(code_size, sample_size)
    aut = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    print(save_weights_path)
    print(load_weights_path)
    if load_weights_path is not None:
        aut.load_state_dict(load_weights_path)

    for ii in range(num_iters):
        print(str(ii) + '/' + str(num_iters) + ' ')
        aut.train(num_sub_steps, batch_size)

    # Save the model
    aut.save_state_dict(save_weights_path)


    # Visualizations
    if do_visualize and code_size >= 3:
        codes = aut.encode(Variable(torch.FloatTensor(dataset))).data.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(codes[:,0],codes[:,1],codes[:,2],c=np.arange(codes.shape[0]))
        plt.show()


    # Try to reconstruct one of the samples
    recon = aut.reconstruct(Variable(torch.FloatTensor(dataset[100:101]))).data.cpu().numpy()
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
    recon = aut.reconstruct(Variable(torch.FloatTensor(dataset[100:101]))).data.cpu().numpy()
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

