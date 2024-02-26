import sys
import numpy as np
import math
import faiss
import os
path = os.path.dirname(np.__file__)
path = path[:-5] + "data_reduction"
path
##############################################################################
#PHL
#PH Landmarks Selection

# sys.path.append(path + "Outlier-robust-subsampling-techniques-for-persistent-homology")
# from getPHLandmarks import getPHLandmarks
# # os.chdir("../../")

from ripser import ripser
from scipy.spatial import KDTree


def getMaxPersistence(ripser_pd):

	if ripser_pd.size == 0:
		max_persistence = 0
	else:
		finite_bars_where = np.invert(np.isinf(ripser_pd[:,1]))
		finite_bars = ripser_pd[np.array(finite_bars_where),:]
		max_persistence = np.max(finite_bars[:,1]-finite_bars[:,0])

	return max_persistence


def getPHOutlierScores_multiDim(point_cloud,topological_radius,max_dim_ripser):

	set_of_super_outliers = np.empty((0,point_cloud.shape[1]))
	super_outlier_indices = np.empty((0,0), int)
	outlier_scores_point_cloud_original_order = np.empty((0,0))

	#for point in point cloud, get delta nhood
	for point_index in range(point_cloud.shape[0]):
		outlier_score = 0

		point = point_cloud[point_index,:]
		point_cloud_minus_point = np.delete(point_cloud,point_index,axis=0)

		kd_tree = KDTree(point_cloud_minus_point)
		indices = kd_tree.query_ball_point(point, r=topological_radius)

		number_of_neighbours = len(indices)


		if number_of_neighbours < 2:

			set_of_super_outliers = np.append(set_of_super_outliers, [point], axis=0)
			super_outlier_indices = np.append(super_outlier_indices,point_index)

		else:

			delta_point_cloud = point_cloud_minus_point[indices,:]

			diagrams = ripser(delta_point_cloud, maxdim=max_dim_ripser)['dgms']

			for dimension in range(max_dim_ripser+1):
				intervals = diagrams[dimension]
				max_persistence = getMaxPersistence(intervals)
				#print max_persistence
				if max_persistence > outlier_score:
					outlier_score = max_persistence


		outlier_scores_point_cloud_original_order = np.append(outlier_scores_point_cloud_original_order,outlier_score)

		#print("This is the outlier score", outlier_score)

	return outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices


def getPHOutlierScores_restrictedDim(point_cloud,topological_radius,dimension):

	set_of_super_outliers = np.empty((0,point_cloud.shape[1]))
	super_outlier_indices = np.empty((0,0), int)
	outlier_scores_point_cloud_original_order = np.empty((0,0))

	#for point in point cloud, get delta nhood
	for point_index in range(point_cloud.shape[0]):
		outlier_score = 0

		point = point_cloud[point_index,:]
		point_cloud_minus_point = np.delete(point_cloud,point_index,axis=0)

		kd_tree = KDTree(point_cloud_minus_point)
		indices = kd_tree.query_ball_point(point, r=topological_radius)

		number_of_neighbours = len(indices)


		if number_of_neighbours < 2:

			set_of_super_outliers = np.append(set_of_super_outliers, [point], axis=0)
			super_outlier_indices = np.append(super_outlier_indices,point_index)

		else:

			delta_point_cloud = point_cloud_minus_point[indices,:]

			diagrams = ripser(delta_point_cloud, maxdim=dimension)['dgms']

			intervals = diagrams[dimension]
			outlier_score = getMaxPersistence(intervals)


		outlier_scores_point_cloud_original_order = np.append(outlier_scores_point_cloud_original_order,outlier_score)

		#print("This is the outlier score", outlier_score)

	return outlier_scores_point_cloud_original_order, set_of_super_outliers, super_outlier_indices


#Landmark function

def getPHLandmarks(point_cloud, topological_radius, sampling_density, scoring_version, dimension, landmark_type):

	number_of_points = point_cloud.shape[0]
	number_of_PH_landmarks = int(round(number_of_points*sampling_density))

	if scoring_version == 'restrictedDim':

		outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices = getPHOutlierScores_restrictedDim(point_cloud,topological_radius,dimension)

	elif scoring_version == 'multiDim':

		max_dim_ripser = dimension
		outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices = getPHOutlierScores_multiDim(point_cloud,topological_radius,max_dim_ripser)


	number_of_super_outliers = set_of_super_outliers.shape[0]

	if landmark_type == 'representative':#small scores

		# sort outlier_scores_point_cloud_original_order
		sorted_indices_point_cloud_original_order = np.argsort(outlier_scores_point_cloud_original_order)

		#Permute zeros
		permuted_super_outlier_indices = np.random.permutation(sorted_indices_point_cloud_original_order[0:number_of_super_outliers])

		sorted_indices_point_cloud_original_order_without_super_outliers = np.array(sorted_indices_point_cloud_original_order[number_of_super_outliers:,])

		# We append the outliers at the end of the vector to select them last
		sorted_indices_point_cloud_original_order = np.append(sorted_indices_point_cloud_original_order_without_super_outliers,permuted_super_outlier_indices)

		PH_landmarks = point_cloud[sorted_indices_point_cloud_original_order[range(number_of_PH_landmarks)],:]

	elif landmark_type == 'vital':#large scores

		# sort outlier_scores_point_cloud_original_order
		sorted_indices_point_cloud_original_order = np.argsort(outlier_scores_point_cloud_original_order)

		#Permute zeros
		permuted_super_outlier_indices = np.random.permutation(sorted_indices_point_cloud_original_order[0:number_of_super_outliers])

		sorted_indices_point_cloud_original_order_without_super_outliers = np.array(sorted_indices_point_cloud_original_order[number_of_super_outliers:,])

		#We append the super outliers before the low scores so we select these last
		sorted_indices_point_cloud_original_order = np.append(permuted_super_outlier_indices,sorted_indices_point_cloud_original_order_without_super_outliers)

		# we flip the vector to keep in line with previous landmark call
		sorted_indices_point_cloud_original_order = np.flip(sorted_indices_point_cloud_original_order)

		PH_landmarks = point_cloud[sorted_indices_point_cloud_original_order[range(number_of_PH_landmarks)],:]


	return PH_landmarks, sorted_indices_point_cloud_original_order, number_of_super_outliers


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

sys.path.append(path + "/Original_repositories/Prototype-Selection-Matrix-Decomposition/code")
from main import rank_samples, reduce_samples

def nrmd_selection(X,y,perc,decomposition='SVD_python'):
    n = math.trunc(len(y)*perc)
    X_sorted, Y_sorted, _ = rank_samples(X=X.T, Y=y, experiment='classification', method=decomposition)
    X_reduced, Y_reduced = reduce_samples(X_sorted=X_sorted,Y_sorted=Y_sorted,n_samples_pick=n)
    X_red = np.copy(X_reduced)
    Y_red = np.copy(Y_reduced[0])
    return X_red.T, Y_red
