'''
K-Means algorithm results.
'''

# libs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

# own libs
from clustering.clustering_functions import *


# load data
data = pd.read_csv("../datasets/pd_speech_features.csv")

# data without class
data_array = data.values
target = data_array[:, -1]
data_array = data_array[:, :-1]

# pre-process data - normalization
normalized_data = StandardScaler().fit_transform(data_array)

# number of clusters selection
k_list = np.arange(2, 20, 2)
cohesions = []
silhouettes = []

# train kmeans with different #clusters and calculate silhouette score
for k in k_list:
    kmeans_obj = KMeans(n_clusters=k).fit(normalized_data)
    cohesions.append(kmeans_obj.inertia_)
    silhouettes.append(metrics.silhouette_score(normalized_data, kmeans_obj.labels_))

plt.figure()
line_chart(plt.gca(), k_list, cohesions, "Cohesion by number of clusters", "#clusters", "Cohesion")
plt.grid()

plt.figure()
line_chart(plt.gca(), k_list, silhouettes, "Silhouette by number of clusters", "#clusters", "Silhouette")
plt.grid()

# fixed kmeans evaluation
kmeans_obj = KMeans(n_clusters=3)
results, unique_labels, labels = evaluate_clustering_alg(kmeans_obj, normalized_data, target)
plot_results(results, kmeans_obj.labels_, unique_labels, "KMeans")
clusters_vis(normalized_data, target, labels, ['c', 'm', 'k'])

plt.show()
