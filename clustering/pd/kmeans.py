"""
K-Means results Parkinson Decease Data set.

Tested pre-processing: correlated variables removing, normalization (StandardScaler),
outlier removing with DBSCAN, PCA for data reduction/transformation
"""

# libs
import pandas as pd
from sklearn.preprocessing import StandardScaler

# own libs
from clustering.clustering_functions import *
from data_exploration.multi_analysis_functions import put_away_vars
from utils import dbscan_outliers_analysis_plot, pca_cumulative_variance_plot


#####
# load data
data = pd.read_csv("../../datasets/pd_speech_features.csv")
target = data.pop('class')

#####
# pre-process data
vars_to_remove = put_away_vars(data.corr(), 0.95)
rvars_names = data.columns[vars_to_remove]
data_array = data.drop(columns=rvars_names).values

normalized_data = StandardScaler().fit_transform(data_array)

dbscan_outliers_analysis_plot(normalized_data, eps_list=[15, 20, 25, 30, 35], min_samples=3)
non_outliers_indexes = DBSCAN(eps=35, min_samples=3).fit(normalized_data).labels_ != -1
data_without_out = normalized_data[non_outliers_indexes, :]
new_target = target[non_outliers_indexes]

pca_obj = pca_cumulative_variance_plot(data_without_out)
first_components = pca_obj.components_[:115]  # aprox 90% variance ratio
reduced_data = np.dot(data_without_out, first_components.T)

#####
# parameter tuning
k_analysis(reduced_data, list(range(2, 20, 2)))

#####
# fixed kmeans evaluation
kmeans_obj = KMeans(n_clusters=2)
results, unique_labels, labels = evaluate_clustering_alg(kmeans_obj, reduced_data, new_target)
plot_results(results, kmeans_obj.labels_, unique_labels, "KMeans")
clusters_vis(reduced_data, new_target, labels, plt.cm.rainbow(np.linspace(0, 1, 2)))

plt.show()
