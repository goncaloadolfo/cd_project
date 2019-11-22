"""
DBSCAN results Parkinson Decease Data Set.

Tested pre-processing: correlated variables removing, normalization with StandardScaller,
data transformation/reduction with PCA.
"""

# libs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# own libs
from clustering.clustering_functions import *
from utils import nearest_nb_distance_plot


#####
# load data
data = pd.read_csv("../../datasets/pd_speech_features.csv")
target = data.pop('class')

#####
# pre-processing
normalized_data = StandardScaler().fit_transform(data.values)
reduced_data = PCA(n_components=115).fit_transform(normalized_data)  # aprox 90% variance ratio

#####
# parameters tuning
nearest_nb_distance_plot(reduced_data)
eps_list = [20, 25, 30]
min_samples_list = [2, 3]
dbscan_parameters(reduced_data, target, eps_list, min_samples_list)

#####
# fixed dbscan evaluation
fixed_dbscan = DBSCAN(eps=30, min_samples=2)
results, unique_labels, labels = evaluate_clustering_alg(fixed_dbscan, reduced_data, target)
plot_results(results, labels, unique_labels, "DBSCAN")
clusters_vis(reduced_data, target, labels, plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))

plt.show()
