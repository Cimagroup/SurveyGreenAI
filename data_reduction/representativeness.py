##############################################################################
#Epsilon search

import numpy as np
from scipy.spatial import cKDTree
def find_epsilon(X,y,X_res,y_res):
    epsilon = 0
    classes = np.unique(y)
    for cl in classes:
        A = X_res[y_res==cl]
        B = X[y==cl]
        kdtree = cKDTree(A)
        epsilon = max(epsilon,max(kdtree.query(B,p=np.inf)[0]))
    return epsilon