

class TerrainRLMotionPreprocessor:
    def __init__(t_feature_idx, x_feature_idx, z_feature_idx):
        self.t_feature_idx = t_feature_idx
        self.x_feature_idx = x_feature_idx
        self.z_feature_idx = z_feature_idx

    def process_clip(clip):
        """
            Args:
                @clip    A 2D numpy array of dimensions (frames, features)
        """
        timesteps = clip[:,:,self.t_feature_idx]

        # convert x and z to relative deltas
        clip[:,:,self.x_feature_idx] = compute_deltas(clip[:,:,self.x_feature_idx])
        clip[:,:,self.z_feature_idx] = compute_deltas(clip[:,:,self.z_feature_idx])

        #TODO: Add rotation invariance (one of the joints should be the root joint)

        # get rid of the timestep
        clip = np.delete(clip, self.t_feature_idx, axis = 2)

        # normalize (leave out the zero padded last frame -> causes inf)
        clip, mean, var = normalize_data(clip)

        return clip

    def normalize_data(clip):
        # flatten the clip to two dimensions in order to compute mean and variance across all frames
        flat_clip = clip.reshape(-1, clip.shape[-1])
        # calculate mean
        mean = np.mean(flat_clip, axis = 0)
        # calculate the variance

    def compute_deltas(arr):
        # compute the one-step deltas in the last dimension
        arr = np.diff(arr, n = 1, axis = -1)

        # nothing to diff the last frame with, so just pad with zeros
        shape = arr.shape
        shape[-1] = 1
        z = np.zeros(shape, dtype = arr.dtype)
        arr = np.concatenate((arr, z), axis = -1)
        return arr

