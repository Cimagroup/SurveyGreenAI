##############################################################
###### This code defines the PH landmark selection function and all its help functions
# Original source: https://github.com/stolzbernadette/Outlier-robust-subsampling-techniques-for-persistent-homology


import numpy as np
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

