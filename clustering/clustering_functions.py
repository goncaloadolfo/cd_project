"""
Clustering functions.
"""

# libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans, DBSCAN

# own libs
from vis_functions import line_chart, bar_chart, heatmap, scatter_plot
from utils import euclidean_distance

# globals
COHESION_KEY = "cohesion"
SEPARATION_KEY = "separation"
SILHOUETTE_KEY = "silhouette"
PURITY_KEY = "purity"
CONTINGENCY_MATRIX_KEY = "contingency_matrix"
RAND_INDEX_KEY = "adjusted_random_index"


#####
# parameter tuning
def k_analysis(data: np.ndarray, k_list: list) -> None:
    cohesions = []
    silhouettes = []

    for k in k_list:
        print("testing k=", k)
        kmeans_obj = KMeans(n_clusters=k).fit(data)
        cohesions.append(kmeans_obj.inertia_)
        silhouettes.append(metrics.silhouette_score(data, kmeans_obj.labels_))

    plt.figure()
    line_chart(plt.gca(), k_list, cohesions, "Cohesion by number of clusters", "#clusters", "Cohesion")
    plt.grid()

    plt.figure()
    line_chart(plt.gca(), k_list, silhouettes, "Silhouette by number of clusters", "#clusters", "Silhouette")
    plt.grid()


def dbscan_parameters(data: np.ndarray, target: np.ndarray, eps_list: list, min_samples_list: list) -> None:
    for eps in eps_list:
        for min_samples in min_samples_list:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(data)
            labels = dbscan.labels_
            unique_labels = np.unique(labels)

            print("eps: " + str(eps) + ", min samples: " + str(min_samples))

            if len(unique_labels) > 1:
                print("clusters: " + str(np.unique(labels)))
                print("silhouette: " + str(metrics.cluster.silhouette_score(data, labels)))
                print("adjusted rand index: " + str(metrics.adjusted_rand_score(target, labels)) + "\n")

            else:
                print("all data points were considered outliers...\n")


######
# auxiliary functions for clustering evaluation
def calculate_centroids(data: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray) -> list:
    return [np.mean(data[labels == label], axis=0) for label in unique_labels]


def neareast_centroid(centroid_i: int, centroids: np.ndarray) -> int:
    centroid = centroids[centroid_i]
    distances = [euclidean_distance(centroid, other_centroid) for other_centroid in centroids]
    sorted_inds = np.argsort(distances)

    for dist_ind in sorted_inds:
        if dist_ind != centroid_i:
            return centroids[dist_ind]


######
# clustering evaluation
def evaluate_clustering_alg(alg_obj, preprocessed_data: np.ndarray, targets: np.ndarray) -> tuple:
    # apply clustering alg and get label of each sample
    alg_obj.fit(preprocessed_data)
    labels = alg_obj.labels_
    unique_labels = np.unique(labels)

    # results per cluster
    cohesions = []
    separations = []
    silhouettes = []
    purity = []
    contingency_matrix = []

    centroids = calculate_centroids(preprocessed_data, labels, unique_labels)
    silhouettes_per_sample = metrics.silhouette_samples(preprocessed_data, labels)

    for label_i in range(len(unique_labels)):
        label = unique_labels[label_i]
        centroid = centroids[label_i]

        # cohesion
        within_distances = [euclidean_distance(cluster_point, centroid)**2
                            for cluster_point in preprocessed_data[labels == label]]
        cohesions.append(np.sum(within_distances))

        # separation
        separations.append(np.linalg.norm(centroid) *
                           (euclidean_distance(centroid, neareast_centroid(label_i, centroids))**2))

        # silhouette
        cluster_silhouettes = silhouettes_per_sample[labels == label]
        silhouettes.append(np.sum(cluster_silhouettes) / len(cluster_silhouettes))

        # purity
        if targets is not None:
            cluster_cm = metrics.cluster.contingency_matrix(targets[labels == label], labels[labels == label])
            purity.append(np.sum(np.max(cluster_cm, axis=0)) / np.sum(cluster_cm))

    if targets is not None:
        contingency_matrix = metrics.cluster.contingency_matrix(targets, labels)

    arandom_index = metrics.cluster.adjusted_rand_score(targets, labels)

    # create and return dict with results per cluster
    results_dict = {
        COHESION_KEY: cohesions,
        SEPARATION_KEY: separations,
        SILHOUETTE_KEY: silhouettes,
        PURITY_KEY: purity,
        CONTINGENCY_MATRIX_KEY: contingency_matrix,
        RAND_INDEX_KEY: arandom_index
    }

    return results_dict, unique_labels, labels


#######
# results plotting
def plot_results(results: dict, labels_per_sample: np.ndarray, cluster_labels: np.ndarray, alg_name: str):
    external_measures = len(results[PURITY_KEY]) != 0

    # create figure
    fig, axs = plt.subplots(2, 2) if not external_measures else plt.subplots(3, 2)
    fig.suptitle(alg_name + " results")
    samples_per_cluster = [np.sum(labels_per_sample == label) for label in cluster_labels]

    # plots
    line_chart(axs[0, 0], cluster_labels.astype(np.str), results[COHESION_KEY], "Cohesion per cluster", "cluster",
               "cohesion")
    line_chart(axs[0, 1], cluster_labels.astype(np.str), results[SEPARATION_KEY], "Separation per cluster", "cluster",
               "separation")
    line_chart(axs[1, 0], cluster_labels.astype(np.str), results[SILHOUETTE_KEY], "Silhouette per cluster", "cluster",
               "silhouette")
    bar_chart(axs[1, 1], cluster_labels.astype(np.str), samples_per_cluster, "Samples per cluster", "cluster",
              "#samples", rotation=0)

    if external_measures:
        heatmap(axs[2, 1], results[CONTINGENCY_MATRIX_KEY], "Adjusted random index=" +
                str(np.round(results[RAND_INDEX_KEY], 4)), "Cluster", "Target", False)
        line_chart(axs[2, 0], cluster_labels.astype(np.str), results[PURITY_KEY], "Purity per cluster", "cluster",
                   "purity")

    fig.tight_layout()


def clusters_vis(data: np.ndarray, target: np.ndarray, cluster_labels: np.ndarray, colors: list):
    # data reduction
    feature_selector = SelectKBest(f_classif, k=2)
    data_2d = feature_selector.fit_transform(data, target)

    fig, ax = plt.subplots(1, 2)
    # ground truth vis
    unique_target = np.unique(target)
    for target_i in range(len(unique_target)):
        data_from_ctarget = data_2d[target == unique_target[target_i]]
        scatter_plot(ax[0], data_from_ctarget[:, 0], data_from_ctarget[:, 1], "GT Scatter",
                     "feature 1", "feature 2", "target " + str(unique_target[target_i]), colors[target_i])
    ax[0].legend()

    # clusters vis
    clusters = np.unique(cluster_labels)
    for cluster_i in range(len(clusters)):
        data_from_ccluster = data_2d[cluster_labels == clusters[cluster_i]]
        scatter_plot(ax[1], data_from_ccluster[:, 0], data_from_ccluster[:, 1], "Clusters",
                     "feature 1", "feature 2", "cluster " + str(clusters[cluster_i]), colors[cluster_i])
    ax[1].legend()
