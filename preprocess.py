import numpy as np
import util

def compute_normalization(data):
    # flatten the data to two dimensions in order to compute mean and variance across all frames
    flat_data = data.reshape(-1, data.shape[-1])
    # calculate mean
    mean = np.mean(flat_data, axis = 0)
    # calculate std
    std = np.std(flat_data, axis = 0)
    return mean, std

def apply_normalization(data, mean, std):
    data = (data - mean) / std
    return data

def compute_deltas(arr):
    # compute the one-step deltas in the last dimension
    arr = np.diff(arr, n = 1, axis = -1)

    # nothing to diff the last frame with, so just pad with zeros
    shape = list(arr.shape)
    shape[-1] = 1
    z = np.zeros(tuple(shape), dtype = arr.dtype)
    arr = np.concatenate((arr, z), axis = -1)
    return arr

def compute_cumsum(arr, constant = 0):
    output = np.zeros(arr.shape)
    output.fill(constant)

    # compute the cumulative sum
    # throw out last value as it will not fit
    arr = np.cumsum(arr)[:-1]

    # add one element over
    output[1:] = output[1:] + arr
    return output

def compute_gradients(arr):
    # compute the 2nd order gradients in the last dimension
    arr = np.gradient(arr, axis = -1)
    return arr



class TerrainRLMotionPreprocessor:
    def __init__(self, t_feature_idx, x_feature_idx, z_feature_idx, window_size, stride):
        self.t_feature_idx = t_feature_idx
        self.x_feature_idx = x_feature_idx
        self.z_feature_idx = z_feature_idx
        self.window_size = window_size
        self.stride = stride
        # TODO: Implement proper time step sampling with linear interpolation
        self.sampling_step = 0.033333

    def generate_dataset(self, files_list):
        """ Given a list of filenames, parses them and does preprocessing to return a usable dataset.
        """
        dataset = None
        for file_path in files_list:
            with open(file_path) as mfile:
                motion_clip = util.parse_file_motion_windows(mfile, None)
                windows = self.process_clip(clip=motion_clip, window_size=self.window_size, stride=self.stride)
                if dataset is None:
                    dataset = windows
                else:
                    dataset = np.concatenate((dataset, windows))

        # normalize dataset
        # self.mean, self.std = compute_normalization(dataset)
        # apply_normalization(dataset, self.mean, self.std)
        return dataset

    def process_clip(self, clip, window_size, stride):
        """
            Args:
                @clip    A 2D numpy array of dimensions (frames, features)
        """
        timesteps = clip[:,self.t_feature_idx]

        # convert x and z to diffs
        clip[:,self.x_feature_idx] = compute_deltas(clip[:,self.x_feature_idx])
        clip[:,self.z_feature_idx] = compute_deltas(clip[:,self.z_feature_idx])

        # throw out last frame since nothing to diff x and z with
        clip = clip[:-1,:]
        #TODO: Add rotation invariance (one of the joints should be the root joint)

        # get rid of the timestep
        clip = np.delete(clip, self.t_feature_idx, axis = -1)

        # convert array to strided windows
        windows = util.stride_windows(clip, window_size=window_size, stride=stride)

        return windows

    def convert_back(self, clip):
        """
            Responsible for converting back a processed frame to TerrainRL motion clip.
            Args:
                @clip    A 2D numpy array of dimensions (frames, features)
        """
        # Apply the inverse normalization
        # print('std', self.std)
        # print(self.mean)
        # clip = clip * self.std + self.mean

        # Insert the timesteps
        timesteps = np.zeros(clip.shape[0])
        timesteps.fill(self.sampling_step)
        clip = np.insert(clip, self.t_feature_idx, timesteps, axis = -1)

        # Integrate the diffs
        clip[:,self.x_feature_idx] = compute_cumsum(clip[:,self.x_feature_idx])
        clip[:,self.z_feature_idx] = compute_cumsum(clip[:,self.z_feature_idx])

        return clip

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


