import numpy as np

def error(Y, Y_pred):
    return (len(Y) - np.count_nonzero(Y == Y_pred))/len(Y)