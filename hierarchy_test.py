#!/usr/bin/env python
from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn import metrics
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


"""
            wine        iris        yeast       red-wine
RSLFC       49.3%       89.3%       3.0%        3.2%
K-Means     44.9%       66.6%       19.3%       4.1%
DBSCAN      0.0%        25.1%       2.6%        0.0%
"""


# data = np.loadtxt("data/iris.data.txt", delimiter=',')
# cluster = np.array(data[:,-1])
# data_set = data[:,:-1]

# data = np.loadtxt("data/wine.data.txt", delimiter=',')
# cluster = np.array(data[:,0], dtype=int)
# data_set = data[:,1:]

data = np.loadtxt("data/yeast.data.txt")
cluster = np.array(data[:,-1])
data_set = data[:,:-1]

# data = np.loadtxt("data/redwine.data.txt", delimiter=';')
# cluster = np.array(data[:,-1])
# data_set = data[:,:-1]


kmeans = KMeans(n_clusters=5, random_state=0).fit(data_set)
print(metrics.homogeneity_score(cluster, kmeans.labels_))

# data_set, _ = make_blobs(1000)
# plt.scatter(data_set.T[0], data_set.T[1], color='b')
# plt.show()

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
cluster_labels = clusterer.fit_predict(data_set)
print(metrics.homogeneity_score(cluster, cluster_labels))

db = DBSCAN(eps=0.3, min_samples=10).fit(data_set)
print(metrics.homogeneity_score(cluster, db.labels_))


# clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.414, approx_min_span_tree=True,
#     gen_min_span_tree=False, leaf_size=40, metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
# clusterer.fit(data_set)

# clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
# plt.show()