'''
Clustering functions.
'''

# libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

# own libs
from vis_functions import line_chart


def kmeans_silhouette_vs_nrclusters(preprocessed_data: np.ndarray, k_list: list):
    silhouettes = []

    # train kmeans with different #clusters and calculate silhouette score
    for k in k_list:
        kmeans_obj = KMeans(n_clusters=k).fit(preprocessed_data)
        labels = kmeans_obj.labels_
        silhouettes.append(metrics.silhouette_score(preprocessed_data, labels))

    # plot
    plt.figure()
    line_chart(plt.gca(), k_list, silhouettes, "Silhouettes by number of clusters", "#clusters", "silhouette")
    plt.grid()
