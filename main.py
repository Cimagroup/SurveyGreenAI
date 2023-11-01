import numpy as np

# =============================================================================
# Method 1: Random Stratified Sampling
# =============================================================================

from sklearn.model_selection import StratifiedShuffleSplit


# =============================================================================
# Method 2: Maxmin selection
# =============================================================================

from maxmin import *

def maxmin_selector(X,y,perc,p=np.inf):
    
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
        return np.sort(picks)
    
# =============================================================================
# Method 3: Instance Ranking by Matrix Decomposition
# =============================================================================



# =============================================================================
# Method 4: Persistent Homology Landmarks
# =============================================================================

from PH_landmark_selection import *

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


# =============================================================================
# Method 5: Representative datasets
# =============================================================================

from representative_dataset import *

def dominatingSet(X,y,epsilon=0.1):
    "Dominating dataset of X with a given labels y and representativeness factor epsilon."
    ady = distance_matrix(X,X,p=np.inf)
    g = nx.from_numpy_matrix(ady<epsilon)
    dom = nx.dominating_set(g)
    return np.array(list(dom))


# =============================================================================
# Method 6: Dataset distillation
# =============================================================================


