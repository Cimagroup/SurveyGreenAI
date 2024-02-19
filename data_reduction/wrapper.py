###################################
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
###################################

def fes_selection(y, current_accuracy, forgetting_events, perc, maximum, X=None):
    classes = np.unique(y)
    indices = np.array([], dtype='int32')
    n_y = len(y)
    
    for cl in classes:
        y_cl = y[y == cl]
        n = len(y_cl)
        indices_cl = np.arange(n_y)[y == cl]
        current_cl = current_accuracy[y == cl]
        forgetting_cl = forgetting_events[y == cl]

        for i in range(n):
            if current_cl[i] == 0 and forgetting_cl[i] == 0:
                forgetting_cl[i] = maximum
        
        n_cl = math.trunc(n * perc)
        indices_selected = indices_cl[np.argsort(forgetting_cl)[-n_cl:]]
        indices = np.concatenate((indices, indices_selected))
    
    if X is not None:
        X_res = X[indices]
        y_res = y[indices]
        return X_res, y_res
    else:
        return indices
