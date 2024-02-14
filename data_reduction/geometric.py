import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from gudhi.subsampling import choose_n_farthest_points

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

    if perc_base > perc:
        raise ValueError("perc_base must be smaller than perc")
    
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