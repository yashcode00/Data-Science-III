import numpy as np
from sklearn import metrics

vectors = [[0,1],[-1,1],[-1,2],[-2,1]]
index = np.argmin(metrics.pairwise_distances(vectors).sum(axis=0))
print(vectors[index])