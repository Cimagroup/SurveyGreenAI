import numpy as np
from scipy.spatial import distance_matrix
import random

def dominatingSet(X,y,epsilon=0.1,p=np.inf):
    
    picks = np.array([],dtype=int) 
    classes = np.unique(y)
    for cl in classes:
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        pool_cl = np.reshape(pool_cl,(-1,))
        picks_cl = pick_from_class(X_cl,pool_cl,epsilon,p)
        picks = np.append(picks,picks_cl)
    return np.sort(picks)

def pick_from_class(X_cl,pool_cl,epsilon=0.1,p=np.inf):
    
    lenpool_cl, dim = np.shape(X_cl)
    if epsilon <=0:
        return pool_cl
    else:        
        picks_cl = np.array([],dtype=int)
        pool_aux = np.arange(0,lenpool_cl)
        while lenpool_cl > 0:
            r = random.randrange(lenpool_cl)
            picks_cl = np.append(picks_cl, pool_cl[r])
            m = distance_matrix(np.reshape(X_cl[pool_aux[r]],(-1,dim)),X_cl[pool_aux],p=p)[0]
            remain = np.where(m>=epsilon)
            pool_aux = pool_aux[remain]
            pool_cl = pool_cl[remain]
            lenpool_cl = len(pool_cl)
        return picks_cl 