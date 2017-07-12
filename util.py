import json
from json import encoder
import os

import numpy as np


# Strided Windows

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


# Parsing Files

def parse_file_motion_windows(mfile, window_size, stride = 1):
    """
        Parses a motion file into windows.

        Args:
            @mfile          A file object from which motion should be added.
            @window_size    Size of windows of motions to return.
                            If None will return the whole clip.
            @stride         Stride between windows of motion to return.

        Returns:
            @samples        A numpy ndarray with the motion windows.
    """
    mdata = json.load(mfile)
    motion = np.array(mdata['Frames'], dtype=np.float32)
    if window_size is None:
        samples = motion
    else:
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

def output_motion(frames, out_path):
    """
        Responsible for writing a TerrainRL motion back to a file.
        Args:
            @frames     A 2D numpy array of dimensions (frames, features) in
                        TerrainRL motion frames format.
    """
    # Construct the TerrainRLMotion json dictionary
    mdata = {}
    mdata['Frames'] = frames.tolist()
    mdata['Loop'] = False
    with open(out_path, 'w') as outfile:
        # TODO: figure out how to format floats properly
        # encoder.FLOAT_REPR = lambda f: ("%.2f" % f)
        json.dump(mdata, outfile)

def file_list_from_dir_list(dirs):
    """
        Generates a list of file paths for the files contained in the list of directories.
    """
    file_paths = []
    for diri in dirs:
        file_paths += [os.path.join(diri, x) for x in os.listdir(diri)]
    return file_paths


