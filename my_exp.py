#!/usr/bin/env python


import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import hdbscan

n_samples = 1500
noisy_circles, _ = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2)

plt.figure()
plt.scatter(no_structure.T[0], no_structure.T[1], color='b')

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
clusterer.fit(no_structure)
print(clusterer.labels_)

palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.figure()
plt.scatter(no_structure.T[0], no_structure.T[1], c=cluster_colors)

plt.show()