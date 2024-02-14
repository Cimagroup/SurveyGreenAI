###################################
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
###################################

# def fes_selection(X,y,current_accuracy, forgetting_events,perc):
#     n_y = len(y)
#     for i in range(n_y):
#         if current_accuracy[i]==0 and forgetting_events[i]==0:
#             forgetting_events[i]=np.inf
#     n = math.trunc(n_y*perc)
#     indices = np.argsort(forgetting_events)[-n:]
#     X_res = X[indices]
#     y_res = y[indices]
#     return X_res, y_res

def fes_selection(X,y,current_accuracy,forgetting_events,perc):
    classes = np.unique(y)
    indices = np.array([],dtype='int32')
    n_y = len(y)
    for cl in classes:
        X_cl = X[y==cl]
        y_cl = y[y==cl]
        n = len(y_cl)
        indices_cl = np.arange(n_y)[y==cl]
        current_cl = current_accuracy[y==cl]
        forgetting_cl = forgetting_events[y==cl]
        for i in range(n):
            if current_cl[i]==0 and forgetting_cl[i]==0:
                forgetting_cl[i]=np.inf
        n_cl = math.trunc(n*perc)
        indices_selected = indices_cl[np.argsort(forgetting_cl)[-n_cl:]]
        indices = np.concatenate((indices,indices_selected))
    X_res = X[indices]
    y_res = y[indices]
    return X_res, y_res