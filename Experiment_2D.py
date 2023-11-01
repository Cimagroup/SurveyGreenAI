import matplotlib.pyplot as plt
from main import *
from sklearn.datasets import make_classification


X,y=make_classification(200, n_features=2, n_redundant=0,n_classes=2,class_sep=1.0)



sm1 = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
m1 = sm1.get_n_splits(X, y) #TBD

m2 = maxmin_selector(X,y,0.2,p=np.inf)

m3 = getPHLandmarks(X, 0.4, 0.2, scoring_version='restrictedDim', dimension=1, landmark_type="representative")

m5 = dominatingSet(X,y,epsilon=0.4)


subset2 = X[m2]
subset3 = m3[0]
subset5 = X[m5]



plt.scatter(X[:,0],X[:,1],c=y)
plt.scatter(subset2[:,0],subset2[:,1],marker='x',s=20,c='r')
plt.title("Maxmin")
plt.show()
plt.scatter(X[:,0],X[:,1],c=y)
plt.scatter(subset3[:,0],subset3[:,1],marker='x',s=20,c='r')
plt.title("PH landmarks")
plt.show()
plt.scatter(X[:,0],X[:,1],c=y)
plt.scatter(subset5[:,0],subset5[:,1],marker='x',s=20,c='r')
plt.title("Representative dataset")
plt.show()
