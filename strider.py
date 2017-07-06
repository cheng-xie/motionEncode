import numpy as np

'''
Creates strided window views over sequence data with desired window length and stride.
Used to generate subclips from long motion clips.
'''
def gen_strided_windows(arr, window_size, stride=1):
    for t in range(0, (arr.shape[0])-window_size+1, stride):
        yield arr[t:t+window_size]

def stride_windows(arr, window_size, stride=1):
    return np.stack(list(gen_strided_windows(arr, window_size, stride)))
