'''
KMeans algorithm results for Cover Type data set.
'''

# libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

# own libs
from clustering.clustering_functions import evaluate_clustering_alg, plot_results
from utils import undersampling_ct
from vis_functions import line_chart


#####
# load data
ct_data = undersampling_ct("../../datasets/secondDataSet.csv")
target = ct_data.pop('Cover_Type')
data_array = ct_data.values

#####
# pre processing
normalized_data = StandardScaler().fit_transform(data_array)

#####
# paramater tuning - comment this if necessary - k discovery
k_list = np.arange(10, 40, 2)
cohesions = []
silhouettes = []

for k in k_list:
    print("currently testing k=" + str(k))
    kmeans_obj = KMeans(n_clusters=k).fit(normalized_data)
    cohesions.append(kmeans_obj.inertia_)
    silhouettes.append(metrics.silhouette_score(normalized_data, kmeans_obj.labels_))

plt.figure()
line_chart(plt.gca(), k_list, cohesions, "Cohesion by number of clusters", "#clusters", "Cohesion")
plt.grid()

plt.figure()
line_chart(plt.gca(), k_list, silhouettes, "Silhouette by number of clusters", "#clusters", "Silhouette")
plt.grid()

#####
# fixed algorithm evaluation
kmeans_obj = KMeans(n_clusters=25)
results, unique_labels, labels = evaluate_clustering_alg(kmeans_obj, normalized_data, target)
plot_results(results, kmeans_obj.labels_, unique_labels, "KMeans")

plt.show()
