import os
import numpy as np
import math
from sklearn.model_selection import StratifiedShuffleSplit

##############################################################################
#SRS
#Stratified Random Selection

def srs_selection(X,y,perc):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=perc, random_state=0)
    for pool, picks in sss.split(X, y):
        X_res, y_res = X[picks], y[picks]
    return X_res, y_res

##############################################################################
#PRD
#ProtoDash

os.chdir("Original_repositories/AIX360")
from aix360.algorithms.protodash import ProtodashExplainer
os.chdir("../../")

def prd_selection(X,y,perc, sigma=2, opt='osqp'):
    
    picks = np.array([],dtype=int)
    classes = np.unique(y)
    for cl in classes:
        
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        pool_cl = np.reshape(pool_cl,(-1,))
        n_cl = math.trunc(len(X_cl)*perc)
        
        explainer = ProtodashExplainer()
        (_, S, _) = explainer.explain(X_cl, X_cl, m=n_cl, sigma=sigma, optimizer=opt)
        picks = np.append(picks,pool_cl[S])
        
    picks = np.sort(picks)
    X_res = X[picks]
    y_res = y[picks]
    return X_res, y_res