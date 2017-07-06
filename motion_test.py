import json
from json import encoder
import strider

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from models.mlpcoder import MLPEncoder, MLPDecoder
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader

def test_overfit_motion(motion_file_path, out_path):
    # preprocessing
    samples = None
    mdata = None
    window_size = 10
    with open(motion_file_path) as mfile:
        mdata = json.load(mfile)
        motion = np.array(mdata['Frames'], dtype=np.float32)
        samples = strider.stride_windows(motion, window_size=window_size, stride=1)

    print(samples.shape)
    # Flatten features as we are just using a MLP
    samples = samples.reshape(samples.shape[0],-1)
    print(samples.shape)

    num_iters = 300
    code_size = 10
    sample_size = samples.shape[1]
    batch_size = samples.shape[0]

    data_loader = DataLoader(torch.FloatTensor(samples), batch_size=batch_size, shuffle=True)

    enc = MLPEncoder(sample_size, code_size)
    dec = MLPDecoder(code_size, sample_size)
    aut = AutoEncoder(enc, dec, data_loader, use_cuda = True)

    for ii in range(num_iters):
        print(str(ii) + '/' + str(num_iters) + ' ')
        aut.train(20, batch_size)

    # Try to reconstruct one of the samples
    recon = aut.reconstruct(Variable(torch.FloatTensor(samples[0:1]))).data.cpu().numpy()
    recon = recon.reshape(recon.shape[0], window_size, -1)
    # Remove unnecessary dimension
    recon = recon.reshape(window_size, -1)
    print(recon)
    print(recon.shape)

    # postprocessing

    # print to file
    mdata['Frames'] = recon.tolist()
    with open(out_path, 'w') as outfile:
        encoder.FLOAT_REPR = lambda o: format(o, '.6f')
        json.dump(mdata, outfile)

