"""
K-Means results Parkinson Decease Data set.

Tested pre-processing: normalization (StandardScaler), outlier removing with DBSCAN,
PCA for data reduction/transformation
"""

# libs
from sklearn.preprocessing import StandardScaler

# own libs
from clustering.clustering_functions import *
from utils import undersampling_ct, dbscan_outliers_analysis_plot, nearest_nb_distance_plot, \
    pca_cumulative_variance_plot


#####
# load data
ct_data = undersampling_ct("../../datasets/secondDataSet.csv")
target = ct_data.pop('Cover_Type')
data_array = ct_data.values

#####
# pre processing
normalized_data = StandardScaler().fit_transform(data_array)

nearest_nb_distance_plot(normalized_data[:8000])
dbscan_outliers_analysis_plot(normalized_data, eps_list=[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7], min_samples=3)
dbscan_obj = DBSCAN(eps=0.7, min_samples=3)
dbscan_obj.fit(normalized_data)
data_without_out = normalized_data[dbscan_obj.labels_ != -1]
new_target = target[dbscan_obj.labels_ != -1]

pca_obj = pca_cumulative_variance_plot(data_without_out)
first_components = pca_obj.components_  # 100% variance ratio
reduced_data = np.dot(data_without_out, first_components.T)

#####
# parameter tuning
k_analysis(reduced_data, list(range(2, 40, 2)))

#####
# fixed algorithm evaluation
kmeans_obj = KMeans(n_clusters=25)
results, unique_labels, labels = evaluate_clustering_alg(kmeans_obj, reduced_data, new_target)
plot_results(results, kmeans_obj.labels_, unique_labels, "KMeans")
clusters_vis(reduced_data, new_target, labels, plt.cm.rainbow(np.linspace(0, 1, 25)))

plt.show()
