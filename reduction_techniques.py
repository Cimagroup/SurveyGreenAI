import sys
import numpy as np
import math
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import faiss
from gudhi.subsampling import choose_n_farthest_points

##############################################################################
#SRS
#Stratified Random Selection

def srs_selection(X,y,perc):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=perc, random_state=0)
    for pool, picks in sss.split(X, y):
        X_res, y_res = X[picks], y[picks]
    return X_res, y_res

##############################################################################
#CLC
#Clustering Centroids Selection

def clc_selection(X,y,perc):
    
    classes = np.unique(y)
    X_res = np.array([],dtype=float)
    y_res = np.array([],dtype=float)
    n_features = np.shape(X)[1]
    
    for cl in classes:
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        n_cl = math.trunc(len(X_cl)*perc)
        kmeans = KMeans(n_clusters=n_cl, random_state=0, init = 'k-means++', n_init="auto").fit(X_cl)
        X_res = np.append(X_res, kmeans.cluster_centers_)
        y_res = np.append(y_res, np.repeat(cl,n_cl))
        
    return X_res.reshape(-1, n_features), y_res

##############################################################################
#MMS
#MaxMin Selection

def mms_selection(X,y,perc):
    if not (0 <= perc <= 1):
        raise ValueError("perc must be a number between 0 and 1")
    else: 
        X_res = np.array([],dtype=float)
        y_res = np.array([],dtype=float)
        n_features = np.shape(X)[1]
        classes = np.unique(y)
        for cl in classes:
            pool_cl = np.where(y==cl)
            X_cl = X[pool_cl]
            n_cl = math.trunc(len(X_cl)*perc)
            mms_cl = np.array(choose_n_farthest_points(points=X_cl,nb_points=n_cl))
            X_res = np.append(X_res, mms_cl)
            y_res = np.append(y_res, np.repeat(cl,n_cl))
        return X_res.reshape(-1, n_features), y_res

##############################################################################
#DES
#Distance-Entropy Selection

def listEuclidean (p1,cen):
    return [euclidean(p1,c) for c in cen]

def euclidean (p1,p2):
    res_e = (p2-p1)**2
    suma = sum(res_e)
    return math.sqrt(suma)

def softmax(distances):
    expon = np.exp(distances)
    suma = np.sum(expon)
    return expon/suma

def entropy(softProb):
    op = softProb*(np.log2(softProb))
    return -1*(np.sum(op))

def des_selection(X, y, perc, perc_base):
    
    #STEP 1: Split training dataset into Base data and Pool Data
    X_Pool,X_Base,y_Pool,y_Base= train_test_split(X, y,stratify=y,test_size=perc_base,random_state=42,shuffle=True)
    
    #STEP 2: Calculate class prototypes in Base Data
    centers=[]
    classes = np.unique(y_Base)
    for cl in classes:
        pool_cl = np.where(y_Base==cl)
        X_cl = X_Base[pool_cl]
        center_cl = np.mean(X_cl, axis=0)
        centers.append(center_cl)
        
    #STEP 3: Calculate the distance-entropy indicator for Pool Data
    entropies = [entropy(softmax(listEuclidean(x,centers))) for x in X_Pool]

    #STEP 4: Order Pool Data by distance-entropy indicator
    adi = math.trunc(perc*len(y)) - len(y_Base)
    entroMaxs = sorted(entropies,reverse=True)[:adi]
    in_add=[]
    for en in entroMaxs:
        in_add.append(np.where(np.array(entropies) == en)[0][0])
        
    X_add=X_Pool[in_add]
    y_add=y_Pool[in_add]
    
    X_res = np.append(X_Base,X_add,axis=0)
    y_res = np.append(y_Base,y_add)
    
    return X_res, y_res

##############################################################################
#DOM
#Dominating Datasets Selection

sys.path.append("Original_repositories/Experiments-Representative-datasets/notebooks")
from dominating import dominating_dataset

# def dom_selection(X,y,epsilon=1):
#     picks = dominating_dataset(X,y,epsilon)
#     X_res = X[picks]
#     y_res = y[picks]
#     return X_res, y_res

def dom_selection(X,y,epsilon=None,perc=None,initial_epsilon=1,max_iter=10):
    if epsilon != None:
        picks = dominating_dataset(X,y,epsilon)
        X_res = X[picks]
        y_res = y[picks]
        return X_res, y_res
    elif perc != None:
        count=0
        desired_perc=perc
        N=len(y)
        current_epsilon = initial_epsilon
        picks = dominating_dataset(X,y,current_epsilon)
        current_perc = len(picks)/N
        if abs(current_perc - desired_perc) <= 0.01:
            pass
        elif current_perc > desired_perc:
            raise ValueError("The initial epsilon is too low")
        else:
            lower_bound=0
            upper_bound=current_epsilon
            while abs(current_perc-desired_perc)>0.01 and count<max_iter:
                count+=1
                current_epsilon=(upper_bound+lower_bound)/2
                picks = dominating_dataset(X,y,current_epsilon)
                current_perc=len(picks)/N
                if current_perc > desired_perc:
                    lower_bound = current_epsilon
                else:
                    upper_bound = current_epsilon
        X_res = X[picks]
        y_res = y[picks]
        return X_res, y_res 
    else:
        raise ValueError("You must give either an epsilon or a perc")

##############################################################################
#PHL
#PH Landmarks Selection

sys.path.append("Original_repositories/Outlier-robust-subsampling-techniques-for-persistent-homology")
from getPHLandmarks import getPHLandmarks

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

sys.path.append("Original_repositories/Prototype-Selection-Matrix-Decomposition/code")
from sampleReduction_Decomposition import SR_MD

def nrmd_selection(X,y,perc,decomposition='SVD_python'):
    
    classes = np.unique(y)
    X_res = np.array([],dtype=float)
    y_res = np.array([],dtype=float)
    n_features = np.shape(X)[1]
    
    for cl in classes:
        pool = np.where(y==cl)
        X_cl = X[pool]
        n_cl = math.trunc(len(X_cl)*perc)
        srmd_cl = SR_MD(np.transpose(X_cl)).find_scores_and_sort(decomposition)[0]
        srmd_cl = srmd_cl.transpose()[0:n_cl]
        X_res = np.append(X_res, srmd_cl)
        y_res = np.append(y_res, np.repeat(cl,n_cl))
    
    return X_res.reshape(-1, n_features), y_res

##############################################################################
#PSA
#Principal Sample Analysis
sys.path.append("Original_repositories/Principal-Sample-Analysis/PSA_code")
from PSA.PSA_main import PSA

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

##############################################################################
#PRD
#ProtoDash

sys.path.append("Original_repositories/AIX360")
from aix360.algorithms.protodash import ProtodashExplainer

def prd_selection(X,y,perc, sigma=2, opt='osqp'):
    
    picks = np.array([],dtype=int)
    classes = np.unique(y)
    for cl in classes:
        
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        pool_cl = np.reshape(pool_cl,(-1,))
        n_cl = math.trunc(len(X_cl)*perc)
        
        explainer = ProtodashExplainer()
        (W, S, _) = explainer.explain(X_cl, X_cl, m=n_cl, sigma=sigma, optimizer=opt)
        picks = np.append(picks,pool_cl[S])
    picks = np.sort(picks)
    X_res = X[picks]
    y_res = y[picks]
    return X_res, y_res
        