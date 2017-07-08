import json
from json import encoder
import os

import numpy as np

def stride_windows(arr, window_size, stride=1):
    """
        Creates strided window views over sequence data with desired window length and stride.
        Used to generate subclips from long motion clips.
        Args:
            @mfile          A file object from which motion should be added.
            @window_size    Size of windows of motions to return.
            @stride         Stride between windows of motion to return.

        Returns:
            @samples        A numpy ndarray with the motion windows.

    """
    return np.stack(list(gen_strided_windows(arr, window_size, stride)))

def gen_strided_windows(arr, window_size, stride=1):
    for t in range(0, (arr.shape[0])-window_size+1, stride):
        yield arr[t:t+window_size]


def parse_file_motion_windows(mfile, window_size, stride = 1):
    """
        Parses a motion file into windows.

        Args:
            @mfile          A file object from which motion should be added.
            @window_size    Size of windows of motions to return.
            @stride         Stride between windows of motion to return.

        Returns:
            @samples        A numpy ndarray with the motion windows.
    """
    mdata = json.load(mfile)
    motion = np.array(mdata['Frames'], dtype=np.float32)
    samples = stride_windows(motion, window_size=window_size, stride=stride)
    return samples

def add_file_motion_windows(mfile, window_size, stride = 1, dataset = None):
    """
        Adds data from the specified motion file to the dataset.

        Args:
            @mfile          A file object from which motion should be added.
            @window_size    Size of windows of motions to return.
            @stride         Stride between windows of motion to return.
            @dataset        A numpy ndarray where motion windows should be appended to.
                            If None will allocate a new array of windows.

        Returns:
            @dataset        A numpy ndarray with the motion windows appended.
    """
    samples = parse_file_motion_windows(mfile, window_size, stride = stride)
    if dataset is None:
        dataset = samples
    else:
        dataset = np.concatenate((dataset, samples))
    return dataset

def add_dir_motion_windows(mfile_dir, window_size, stride = 1, dataset = None):
    """
        Adds data from motion files in the given file directory to the dataset.

        Args:
            @m_file_dir A string with the path to the dir from which motion should be added.
            @window_size    Size of windows of motions to return.
            @stride         Stride between windows of motion to return.
            @dataset    A numpy ndarray where motion windows should be appended to.
                        If None will allocate a new array of windows.

        Returns:
            @dataset    A numpy ndarray with the motion windows appended.
    """
    for mfile_path in os.listdir(mfile_dir):
        with open(os.path.join(mfile_dir, mfile_path)) as mfile:
            dataset = add_file_motion_windows(mfile, window_size, stride = stride, dataset = dataset)
    return dataset

