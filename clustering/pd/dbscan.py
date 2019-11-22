"""
DBScan algorithm results with default pre-processing.

normalization with StandardScaler -> feature selection with PCA
"""

import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import StandardScaler

from clustering.clustering_functions import *

# load data
data, X, y = pd.read_csv("../../datasets/pd_speech_features.csv")

# pre-processing: normalization
normalized_data = StandardScaler().fit_transform(X)
reduced_data = PCA(n_components=115).fit_transform(normalized_data)  # aprox 90% variance ratio

# nearest neighbour distance
nr_samples = reduced_data.shape[0]
distances = distance_matrix(reduced_data, reduced_data, p=2)
identity_matrix = np.identity(nr_samples, dtype=bool)
identity_matrix = ~identity_matrix
distances_without_diagonal = distances[identity_matrix].reshape((nr_samples, nr_samples - 1))
nn_distance = np.min(distances_without_diagonal, axis=1)

plt.figure()
line_chart(plt.gca(), np.arange(nr_samples), nn_distance, "Nearest neighbour distance", "data point", "distance")

# parameters tuning
eps_list = [15, 20, 25, 30]
min_samples_list = [2, 3, 4]

for eps in eps_list:
    for min_samples in min_samples_list:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(reduced_data)
        labels = dbscan.labels_

        print("eps: ", eps, ", min samples: ", min_samples)
        print("clusters: ", np.unique(labels))
        print("silhouette: ", silhouette_score(reduced_data, labels), "\n")

# fixed dbscan evaluation
fixed_dbscan = DBSCAN(eps=30, min_samples=2)
results, unique_labels, labels = evaluate_clustering_alg(fixed_dbscan, reduced_data, y)
plot_results(results, labels, unique_labels, "DBSCAN")
clusters_vis(reduced_data, y, labels, ['c', 'm', 'y', 'k', 'b'])

plt.show()
