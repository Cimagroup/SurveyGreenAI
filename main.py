import sys
import numpy
import random

#Import methods
from reduction_techniques import srs_selection
from reduction_techniques import clc_selection
from reduction_techniques import mms_selection
from reduction_techniques import des_selection
from reduction_techniques import dom_selection
from reduction_techniques import phl_selection
from reduction_techniques import nrmd_selection
from reduction_techniques import psa_selection
from reduction_techniques import prd_selection

def data_reduction(X, y, method, perc=1, epsilon=None, perc_base=1, initial_epsilon=1, max_iter=10, topological_radius=1, scoring_version='multiDim', dimension=1, landmark_type='vital', decomposition='SVD_python', RANSAC=20, sigma=2, opt='osqp'):

    if method=='None':
        X_res, y_res = X, y
    elif method=='SRS':
        X_res, y_res = srs_selection(X,y,perc)
    elif method=='CLC':
        X_res, y_res = clc_selection(X,y,perc)
    elif method=='MMS':
        X_res, y_res = mms_selection(X,y,perc)
    elif method=='DES':
        X_res, y_res = des_selection(X,y,perc,perc_base)
    elif method=='DOM':
        X_res, y_res = dom_selection(X,y,epsilon,perc,initial_epsilon,max_iter)
    elif method=='PHL':
        X_res, y_res = phl_selection(X, y, topological_radius, perc, scoring_version, dimension, landmark_type)
    elif method=='NRMD':
        X_res, y_res = nrmd_selection(X,y,perc,decomposition)
    elif method=='PSA':
        X_res, y_res = psa_selection(X,y,perc,RANSAC)
    elif method=='PRD':
        X_res, y_res = prd_selection(X,y,perc, sigma, opt)
    else:
        raise ValueError("Invalid method. Valid methods are: None, SRS, CLC, MMS, DES, DOM, PHL, NRMD, PSA, PRD")

    return X_res, y_res
        
    
        
    