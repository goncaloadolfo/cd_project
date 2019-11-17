'''
K-Means results with default pre-processing.

normalization with StandardScaler -> outlier removing with DBSCAN -> feature selection with PCA
'''

# libs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# own libs
from clustering.clustering_functions import *
from vis_functions import line_chart


# load data
data = pd.read_csv("../../datasets/pd_speech_features.csv")
data_array = data.values
target = data_array[:, -1]
data_array = data_array[:, :-1]

# pre-process data - outlier removing, normalization and feature selection
normalized_data = StandardScaler().fit_transform(data_array)

eps_list = [15, 20, 25, 30, 35]
outliers_found = []
for eps in eps_list:
    dbscan_obj = DBSCAN(eps=eps, min_samples=3)
    dbscan_obj.fit(normalized_data)
    outliers_found.append(np.sum(dbscan_obj.labels_ == -1))

plt.figure()
line_chart(plt.gca(), eps_list, outliers_found, "Outliers found per eps used", "eps", "#outliers")
non_outliers_indexes = DBSCAN(eps=30, min_samples=3).fit(normalized_data).labels_ != -1
data_without_out = normalized_data[non_outliers_indexes, :]
new_target = target[non_outliers_indexes]

pca_obj = PCA(n_components=data_without_out.shape[1])
pca_obj.fit(data_without_out, target)
explained_variance_ratio = pca_obj.explained_variance_ratio_
variance_ratio_cumsum = np.cumsum(explained_variance_ratio)

plt.figure()
line_chart(plt.gca(), np.arange(len(explained_variance_ratio)), variance_ratio_cumsum,
           "Cumulative variance ratio in PC", "principal component", "cumulative variance ratio")

first_components = pca_obj.components_[:115]  # aprox 90% variance ratio
reduced_data = np.dot(data_without_out, first_components.T)

# parameter tuning
k_list = np.arange(2, 20, 2)
cohesions = []
silhouettes = []

for k in k_list:
    kmeans_obj = KMeans(n_clusters=k).fit(reduced_data)
    cohesions.append(kmeans_obj.inertia_)
    silhouettes.append(metrics.silhouette_score(reduced_data, kmeans_obj.labels_))

plt.figure()
line_chart(plt.gca(), k_list, cohesions, "Cohesion by number of clusters", "#clusters", "Cohesion")
plt.grid()

plt.figure()
line_chart(plt.gca(), k_list, silhouettes, "Silhouette by number of clusters", "#clusters", "Silhouette")
plt.grid()

# fixed kmeans evaluation
kmeans_obj = KMeans(n_clusters=2)
results, unique_labels, labels = evaluate_clustering_alg(kmeans_obj, reduced_data, new_target)
plot_results(results, kmeans_obj.labels_, unique_labels, "KMeans")
clusters_vis(reduced_data, new_target, labels, ['c', 'm'])

plt.show()
