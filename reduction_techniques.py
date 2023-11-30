import sys
import numpy as np
import math
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

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
#Clustering Centroids Selection (Hecho por mí)

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
#MaxMin Selection (Hecho por mí basado en una charla, que no recuerdo ahora mismo)

def mms_selection(X,y,perc,p=np.inf):
    
    if not (0 <= perc <= 1):
        raise ValueError("perc must be a number between 0 and 1")
    else:
        picks = np.array([],dtype=int) 
        classes = np.unique(y)
        for cl in classes:
            pool_cl = np.where(y==cl)
            X_cl = X[pool_cl]
            pool_cl = np.reshape(pool_cl,(-1,))
            picks_cl = maxmin_from_class(X_cl,pool_cl,n=len(pool_cl)*perc,p=p)
            picks = np.append(picks,picks_cl)
        picks = np.sort(picks)
        X_res = X[picks]
        y_res = y[picks]
        return X_res, y_res
    
    
def maxmin_from_class(data, pool_cl, n, p=np.inf):
    
    if n <= 0:
        raise ValueError("Error: n must be a positive number")
    else:
        lendata = len(pool_cl)
        if n >= lendata:
            return pool_cl
        else:
            pool=np.arange(lendata)
            r=random.randrange(lendata)
            picks= np.array([r],dtype=int)
            pool=np.delete(pool,r)
            picked=1
            distmat=np.array([[r,pool[0],np.linalg.norm(np.subtract(data[picks[0]],data[pool[0]]),ord=p)]])
            for i in range(1,lendata-1):
                distmat = np.vstack((distmat,np.array([[r,pool[i],np.linalg.norm(np.subtract(data[picks[0]],data[pool[i]]),ord=p)]])))
            while picked < n:
                picks, pool, distmat = pick_one(picks,pool,data,distmat,picked,p)
                picked+=1
            return np.sort(pool_cl[picks])

def pick_one(picks,pool,data,distmat,picked,p):
                    
    n_pool=len(pool)
    current_max=0
    current_index=-1
    for i in range(n_pool):
        current_max, current_index, distmat = min_search(i,picks,pool,data,distmat,picked,current_max,current_index,p)
    picks = np.append(picks, pool[current_index])
    distmat = distmat[distmat[:,1]!=pool[current_index]]
    pool = np.delete(pool, current_index)
    return picks, pool, distmat

def min_search(i,picks,pool,data,distmat,picked,current_max,current_index,p):
                    
    current_min = np.inf
    pooli = pool[i]
    for j in range(picked):
        d=np.inf
        picksj = picks[j]
        filt = distmat[(distmat[:, 0] == picksj) & (distmat[:, 1] == pooli)]
        if np.any(filt):
            d=filt[0][2]
        else:
            d=np.linalg.norm(np.subtract(data[picksj],data[pooli]),ord=p)
            distmat=np.vstack((distmat,np.array([[picksj,pooli,d]])))
        if d < current_max:
            return current_max, current_index, distmat
        elif d < current_min:
            current_min = d
    current_max = current_min
    current_index = i
    return current_max, current_index, distmat

##############################################################################
#DES
#Distance-Entropy Selection (Basado en el de Ricardo Peralta, pero con retoques pequeños)

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
    centros=[]
    val, cont = np.unique(y_Base,return_counts=True)
    for i in val:
        ind_i=np.where(y_Base==i)
        suma = sum(X_Base[ind_i])
        centro=suma/cont[i]
        centros.append(centro)
     
    #STEP 3: Calculate the distance-entropy indicator for Pool Data
    entropies = [entropy(softmax(listEuclidean(x,centros))) for x in X_Pool]
    
    #STEP 4: Order Pool Data by distance-entropy indicator
    adi = math.trunc(perc*len(y)) - len(y_Base)
    entroMaxs = sorted(entropies,reverse=True)[:adi]
    in_add=[]
    for en in entroMaxs:
        in_add.append(np.where(np.array(entropies) == en)[0][0])
        
    X_adicionantes=X_Pool[in_add]
    y_adicionantes=y_Pool[in_add]
    
    X_res = np.append(X_Base,X_adicionantes,axis=0)
    y_res = np.append(y_Base,y_adicionantes)
    
    return X_res, y_res

##############################################################################
#DOM
#Dominating Datasets Selection (Esto usa mi código, no el del github del grupo)

def dom_selection(X,y,epsilon=0.1,p=np.inf):
    
    picks = np.array([],dtype=int) 
    classes = np.unique(y)
    for cl in classes:
        pool_cl = np.where(y==cl)
        X_cl = X[pool_cl]
        pool_cl = np.reshape(pool_cl,(-1,))
        picks_cl = pick_from_class(X_cl,pool_cl,epsilon,p)
        picks = np.append(picks,picks_cl)
    picks = np.sort(picks)
    X_res = X[picks]
    y_res = y[picks]
    return X_res, y_res

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

##############################################################################
#PHL
#PH Landmarks Selection

sys.path.append("Original_repositories/Outlier-robust-subsampling-techniques-for-persistent-homology-main")
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

sys.path.append("Original_repositories/Prototype-Selection-Matrix-Decomposition-master/code")
from sampleReduction_Decomposition import SR_MD

def nrmd_selection(X,y,perc,method='SVD_python'):
    
    classes = np.unique(y)
    X_res = np.array([],dtype=float)
    y_res = np.array([],dtype=float)
    n_features = np.shape(X)[1]
    
    for cl in classes:
        pool = np.where(y==cl)
        X_cl = X[pool]
        n_cl = math.trunc(len(X_cl)*perc)
        srmd_cl = SR_MD(np.transpose(X_cl)).find_scores_and_sort(method)[0]
        srmd_cl = srmd_cl.transpose()[0:n_cl]
        X_res = np.append(X_res, srmd_cl)
        y_res = np.append(y_res, np.repeat(cl,n_cl))
    
    return X_res.reshape(-1, n_features), y_res

##############################################################################
#PSA
#Principal Sample Analysis
sys.path.append("Original_repositories/Principal-Sample-Analysis-master/PSA_code")
from PSA.PSA_main import PSA

def psa_selection(X,y,perc,RANSAC):
    
    psa = PSA(save_ranks=False, add_noisy_samples_if_not_necessary=False, number_of_added_noisy_samples=0,
                 number_of_selected_samples_in_group=10, number_of_iterations_of_RANSAC=RANSAC)
    ranks = psa.rank_samples(X,y,demo_mode=False, report_steps=False, HavingClasses=True)
    sorted_samples = psa.sort_samples_according_to_ranks(X, y, ranks)
    labels = list(set(y))
    n_samples = y_res = np.array([],dtype=int)
    for l in labels:
        n_samples = np.append(n_samples,int(np.count_nonzero(y == l)*perc)) 
    
    X_res, y_res = psa.reduce_data(sorted_samples,n_samples)
    return X_res, y_res