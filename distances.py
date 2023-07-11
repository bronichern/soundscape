import numpy as np
from scipy.stats import zscore
from linmdtw import linmdtw
from librosa.sequence import dtw as lib_dtw

def l2(x, y): return np.linalg.norm(x - y)


def mut_normalize_sequences(sq1, sq2, normalize):
    if normalize:
        sq1 = np.copy(sq1)
        sq2 = np.copy(sq2)
        len_sq1 = sq1.shape[0]

        arr = np.concatenate((sq1, sq2), axis=0)
        for dim in range(sq1.shape[1]):
            arr[:, dim] = zscore(arr[:, dim])
        sq1 = arr[:len_sq1, :]
        sq2 = arr[len_sq1:, :]
    return sq1, sq2


def librosa_dtw(sq1, sq2):
    return lib_dtw(sq1.transpose(), sq2.transpose())[0][-1, -1]


def parallel_dtw(t1, t2):
    return linmdtw(t1, t2)
