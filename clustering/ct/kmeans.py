"""
KMeans algorithm results for Cover Type data set.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from clustering.clustering_functions import evaluate_clustering_alg, plot_results
from utils import load_and_undersample_ct
from vis_functions import line_chart

# load data
ct_data = load_and_undersample_ct("../../datasets/secondDataSet.csv")
y = ct_data.pop('Cover_Type')
X = ct_data.values

# pre processing
normalized_data = StandardScaler().fit_transform(X)

# paramater tuning - comment this if necessary - k discovery
k_list = np.arange(10, 40, 2)
cohesions = []
silhouettes = []

for k in k_list:
    print("currently testing k=", k)
    kmeans_obj = KMeans(n_clusters=k).fit(normalized_data)
    cohesions.append(kmeans_obj.inertia_)
    silhouettes.append(metrics.silhouette_score(normalized_data, kmeans_obj.labels_))

plt.figure()
line_chart(plt.gca(), k_list, cohesions, "Cohesion by number of clusters", "#clusters", "Cohesion")
plt.grid()

plt.figure()
line_chart(plt.gca(), k_list, silhouettes, "Silhouette by number of clusters", "#clusters", "Silhouette")
plt.grid()

# fixed algorithm evaluation
kmeans_obj = KMeans(n_clusters=25)
results, unique_labels, labels = evaluate_clustering_alg(kmeans_obj, normalized_data, y)
plot_results(results, kmeans_obj.labels_, unique_labels, "KMeans")

plt.show()
