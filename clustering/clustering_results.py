'''
Clustering results.
'''

# libs
from sklearn.preprocessing import StandardScaler
import pandas as pd

# own libs
from clustering.clustering_functions import *


data = pd.read_csv("../datasets/pd_speech_features.csv")

# data without class
data_array = data.values
target = data_array[:, -1]
data_array = data_array[:, :-1]

# pre-process data - normalization
normalized_data = StandardScaler().fit_transform(data_array)
    
#####
# K-Means

# select a number of clusters for kmeans
k_list = np.arange(2, 20, 2)
kmeans_cohesion_vs_nrclusters(normalized_data, k_list)

# fixed kmeans evaluation
kmeans_obj = KMeans(n_clusters=3)
results, cluster_labels = evaluate_clustering_alg(kmeans_obj, normalized_data, target)
plot_results(results, kmeans_obj.labels_, cluster_labels, "KMeans")

plt.show()
