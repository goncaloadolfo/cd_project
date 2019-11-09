'''
Pattern mining, clustering results.
'''

# libs
from sklearn.feature_selection import f_classif

# own libs
from pattern_mining import *
from clustering import *
from utils import print_return_variable
from sklearn.preprocessing import StandardScaler


def pattern_mining(data: pd.DataFrame):
    y = data["class"].values

    # pattern mining parameters
    print("\n### Pattern mining parameters")
    k_features = print_return_variable("Number of features to select: ", 35)
    selection_measure = print_return_variable("Feature selection function: ", f_classif)
    discretize_function = print_return_variable("Discretize function: ", pd.cut)
    bins = print_return_variable("Number of bins: ", 7)
    disc_needless_cols = print_return_variable("Variables that doesnt need to be discretize: ", ['gender', 'class'])
    min_supp = print_return_variable("Min support: ", 0.6)
    fp_mining_args = [min_supp]
    min_conf = print_return_variable("Min confidence: ", 0.9)
    min_ant_items = print_return_variable("Min of items in antecedents itemset: ", 2)

    # get results
    selected_features, dummified_df, frequent_patterns, _, rules = pm_system(data, y, k_features, selection_measure,
                                                                             discretize_function, bins,
                                                                             disc_needless_cols, fp_mining_args,
                                                                             min_conf, min_ant_items)

    # results visualization
    print("\n### Selected features")
    print("Selected features: ", selected_features)
    print("\n### Frequent patterns found (some)")
    pretty_print_fsets(frequent_patterns, False, 5)
    print("\n### Association rules found (some)")
    pretty_print_rules(rules, ['confidence'], False, 10)

    fsets_per_supp(dummified_df, np.arange(0.4, 1.001, 0.1))


def clustering(data):
    # data without class
    data_array = data.values
    data_array = data_array[:, :-1]

    # pre-process data - normalization
    normalized_data = StandardScaler().fit_transform(data_array)

    #####
    # K-Means

    # select a number of clusters for kmeans
    k_list = np.arange(2, 21)
    kmeans_silhouette_vs_nrclusters(normalized_data, k_list)
    plt.show()


def main():
    data = pd.read_csv("pd_speech_features.csv")
    # pattern_mining(data)
    clustering(data)


if __name__ == "__main__":
    main()
