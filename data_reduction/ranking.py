import os
import numpy as np
import math
import faiss

##############################################################################
#PHL
#PH Landmarks Selection

os.chdir("Original_repositories/Outlier-robust-subsampling-techniques-for-persistent-homology")
from getPHLandmarks import getPHLandmarks
os.chdir("../../")

def phl_selection(X, y, topological_radius, perc, scoring_version, dimension, landmark_type):
    
    classes = np.unique(y)
    X_res = np.array([],dtype=float)
    y_res = np.array([],dtype=float)
    n_features = np.shape(X)[1]
    
    for cl in classes:
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        PHLandmarks = getPHLandmarks(point_cloud=X_cl, topological_radius=topological_radius, sampling_density=perc,  scoring_version=scoring_version, dimension=dimension, landmark_type=landmark_type)[0]
        n_cl = len(PHLandmarks)
        X_res = np.append(X_res, PHLandmarks)
        y_res = np.append(y_res, np.repeat(cl,n_cl))
        
    return X_res.reshape(-1, n_features), y_res

##############################################################################
#NRMD
#Numerosity Reduction with Matrix Decomposition

os.chdir("Original_repositories/Prototype-Selection-Matrix-Decomposition/code")
from main import rank_samples, reduce_samples
os.chdir("../../../")

def nrmd_selection(X,y,perc,decomposition='SVD_python'):
    n = math.trunc(len(y)*perc)
    X_sorted, Y_sorted, _ = rank_samples(X=X.T, Y=y, experiment='classification', method=decomposition)
    X_reduced, Y_reduced = reduce_samples(X_sorted=X_sorted,Y_sorted=Y_sorted,n_samples_pick=n)
    return X_reduced.T, Y_reduced[0]

##############################################################################
#PSA
#Principal Sample Analysis

os.chdir("Original_repositories/Principal-Sample-Analysis/PSA_code")
from PSA.PSA_main import PSA
os.chdir("../../../")

def psa_selection(X,y,perc,RANSAC):
    
    psa = PSA(save_ranks=False, add_noisy_samples_if_not_necessary=False, number_of_added_noisy_samples=0,
                 number_of_selected_samples_in_group=X.shape[1], number_of_iterations_of_RANSAC=RANSAC)
    ranks = psa.rank_samples(X,y,demo_mode=False, report_steps=False, HavingClasses=True)
    sorted_samples = psa.sort_samples_according_to_ranks(X, y, ranks)
    labels = list(set(y))
    n_samples = y_res = np.array([],dtype=int)
    for l in labels:
        n_samples = np.append(n_samples,int(np.count_nonzero(y == l)*perc)) 
    
    X_res, y_res = psa.reduce_data(sorted_samples,n_samples)
    return X_res, y_res
