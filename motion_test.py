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

class TerrainRLMotionEncodeScenario:
    """
        Wrapper for a TerrainRL motion encoder for supervised training.
    """
    def __init__(self, code_dims, sample_dims, preprocessor, use_cuda = True):
        self._code_dims = code_dims
        self._sample_dims = sample_dims
        self._model = self.init_model(sample_dims, code_dims, use_cuda)
        self.preprocessor = preprocessor
        pass

    def init_model(self, sample_dims, code_dims, use_cuda = True):
        enc = MLPEncoder(sample_dims, code_dims)
        dec = MLPDecoder(code_dims, sample_dims)
        model = AutoEncoder(enc, dec, use_cuda)
        return model

    def save_model_param_dict(self, path):
        self._model.save_state_dict(path)

    def load_model_param_dict(self, path):
        self._model.load_state_dict(path)

    def train(self, dataset, epochs, batch_size=-1, iters_per_log = 1):
        """
            Trains the encoding model on the supplied dataset.
            Args:
                @dataset:   A 2D numpy array of dimensions
        """
        assert dataset.shape[1] == self._sample_dims, \
            "Dataset feature dims (%r) incompatible with model (%r)." \
            % (dataset.shape[1], self._sample_dims)

        if batch_size == -1:
            batch_size = dataset.shape[0]

        self._model.train(dataset, epochs, batch_size, iters_per_log)

    def reconstruct(self, clip, window_size):
        """
            Reconstructs an input clip using our model.
        """
        # Try to reconstruct one of the samples
        recon = self._model.reconstruct(clip).data.cpu().numpy()
        recon = recon.reshape(recon.shape[0], window_size, -1)
        recon = recon.reshape(window_size, -1)

        recon = self.preprocessor.convert_back(recon)

        return recon

    def visualize_encoding(self, dataset):
        # Visualizations
        if self._code_dims >= 3:
            codes = self._model.encode(dataset).data.cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(codes[:,0],codes[:,1],codes[:,2],c=np.arange(codes.shape[0]))
            plt.show()

    def generate_dataset(self, file_paths, window_size, stride):
        # Preprocessing
        # TODO: handle different target frame rates
        dataset = None

        # Load data
        preprocessor = Preprocessor(t_feature_idx=0, x_feature_idx=1, z_feature_idx=3, window_size=window_size, stride=stride)
        dataset = preprocessor.generate_dataset(file_paths)

        # Flatten features as we are just using a MLP
        dataset = dataset.reshape(dataset.shape[0],-1)
        return dataset

    def decode(self, code, window_size):
        sample = self._model.decode(code).data.cpu().numpy()
        sample = sample.reshape(sample.shape[0], window_size, -1)
        sample = sample.reshape(window_size, -1)
        return self.preprocessor.convert_back(sample)


def simple_test_multimodal_motion(mfile_dirs, out_path, save_weights_path, load_weights_path = None):
    # Params
    window_size = 10
    stride = 2
    do_visualize = True

    use_cuda = True
    epochs = 1000
    batch_size = -1     # Full batches
    iters_per_log = 10

    code_dims = 3

    # Generate Dataset
    file_paths = util.file_list_from_dir_list(mfile_dirs)
    preprocessor = Preprocessor(t_feature_idx=0, x_feature_idx=1, z_feature_idx=3, \
                                window_size=window_size, stride=stride)
    dataset = preprocessor.generate_dataset(file_paths)
    dataset = dataset.reshape(dataset.shape[0],-1)

    # Create Scenario
    sample_dims = dataset.shape[1]
    scenario = TerrainRLMotionEncodeScenario(code_dims, sample_dims, preprocessor, use_cuda)

    # Load Model
    if load_weights_path is not None:
        scenario.load_model_param_dict(load_weights_path)

    # Train
    scenario.train(dataset, epochs, batch_size, iters_per_log)

    # Save Model
    if save_weights_path is not None:
        scenario.save_model_param_dict(save_weights_path)

    # Visualize
    if do_visualize:
        scenario.visualize_encoding(dataset)

    # Output Reconstruction
    recon = scenario.reconstruct(dataset[100:101], window_size)
    print(recon)
    print(recon.shape)
    util.output_motion(recon, out_path)

    # Try to interpolate between walking and running
    # decoding = scenario.decode(np.array([[0.71,-0.00,0.15]]), window_size)
    # util.output_motion(decoding, out_path)


