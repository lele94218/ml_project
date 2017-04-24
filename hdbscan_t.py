# #!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data

import hdbscan

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)

print(clusterer.labels_)
print(clusterer.probabilities_)
print(clusterer.cluster_persistence_)

plt.figure()
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)

plt.figure()
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

plt.figure()
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
# plt.show()
