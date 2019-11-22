"""
Auxiliary functions to other modules.
"""
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, OneHotEncoder

from vis_functions import line_chart


def print_return_variable(prefix, value):
    print(prefix + str(value))
    return value


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def correlation_analysis_list(data, predicate):
    class Correlated:
        def __init__(self, variable1, variable2, correlation):
            self.variable1 = variable1
            self.variable2 = variable2
            self.correlation = correlation

        def __str__(self):
            return '(%s, %s) = %.4f' % (self.variable1, self.variable2, self.correlation)

        def __unicode__(self):
            return u'(%s, %s) = %.4f' % (self.variable1, self.variable2, self.correlation)

        def __repr__(self):
            return '(%s, %s) = %.4f' % (self.variable1, self.variable2, self.correlation)

    columns = data.select_dtypes(include='number').columns
    corr_mtx = data.corr()
    size = len(columns)
    res = []
    for i in range(0, size):
        for j in range(i + 1, size):
            v = corr_mtx.iloc[i, j]
            if predicate(v):
                res.append(Correlated(columns[i], columns[j], v))
    return res


def remove_correlated_vars(data: pd.DataFrame, predicate) -> pd.DataFrame:
    corr_vars = correlation_analysis_list(data, predicate)
    to_drop = list(map(lambda v: v.variable2, corr_vars))
    return data.drop(columns=to_drop)


def load_pd(pd_path, pop_class=True, remove_corr=False, corr_threshold=.9, merge_observations=False):
    global X, y
    data: pd.DataFrame = pd.read_csv(pd_path)

    if merge_observations:
        i, j = 0, 3
        new_data = pd.DataFrame(columns=data.columns.values)
        while i < data.shape[0]:
            x = data.iloc[i:j]
            mean = x.mean()
            new_data.loc[int(i / 3)] = mean
            i += 3
            j += 3
        new_data.astype({'class': int})
        data = new_data

    if pop_class:
        y = data.pop('class')
        X = data.values

    if remove_corr:
        predicate = lambda v: v > corr_threshold or v < -corr_threshold
        data = remove_correlated_vars(data, predicate)

    if pop_class:
        return data, X, y
    else:
        return data


def load_and_undersample_ct(ct_path):
    # load data and target
    data = pd.read_csv(ct_path)
    target = data['Cover_Type']

    # possible cover types
    cover_types = np.unique(target)

    # area binary column names
    area_col_names = ['Wilderness_Area0', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3']

    # under sampling process
    undersampled_dfs = []
    for cover_type in cover_types:
        for area in area_col_names:
            ct_area_df = data[np.logical_and(data['Cover_Type'] == cover_type, data[area] == 1)]
            nsamples = ct_area_df.shape[0]

            if 0 < nsamples < 4000:
                undersampled_dfs.append(ct_area_df)

            elif nsamples > 0:
                ct_area_df.sample(frac=1)  # shuffle rows
                undersampled_dfs.append(ct_area_df.iloc[:3000])

    # concatenate dfs
    return pd.concat(undersampled_dfs)


def dbscan_outliers_analysis_plot(data: np.ndarray, eps_list: list, min_samples: int) -> None:
    outliers_found = []

    for eps in eps_list:
        print("getting outliers with eps=", eps)
        dbscan_obj = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_obj.fit(data)
        outliers_found.append(np.sum(dbscan_obj.labels_ == -1))

    plt.figure()
    line_chart(plt.gca(), eps_list, outliers_found, "Outliers found per eps used", "eps", "#outliers")


def pca_cumulative_variance_plot(data: np.ndarray) -> PCA:
    pca_obj = PCA(n_components=min(data.shape[0], data.shape[1]))
    pca_obj.fit(data)
    explained_variance_ratio = pca_obj.explained_variance_ratio_
    variance_ratio_cumsum = np.cumsum(explained_variance_ratio)

    plt.figure()
    line_chart(plt.gca(), np.arange(len(explained_variance_ratio)), variance_ratio_cumsum,
               "Cumulative variance ratio in PC", "principal component", "cumulative variance ratio")

    return pca_obj


def nearest_nb_distance_plot(data: np.ndarray):
    nr_samples = data.shape[0]
    distances = distance_matrix(data, data, p=2)

    identity_matrix = np.identity(nr_samples, dtype=bool)
    identity_matrix = ~identity_matrix
    distances_without_diagonal = distances[identity_matrix].reshape((nr_samples, nr_samples - 1))
    nn_distance = np.min(distances_without_diagonal, axis=1)

    plt.figure()
    line_chart(plt.gca(), np.arange(nr_samples), nn_distance, "Nearest neighbour distance", "data point", "distance")


def get_class_balance(data):
    unbal = copy(data)
    target_count = unbal['class'].value_counts()
    # plt.figure()
    # plt.title('Class balance')
    # plt.bar(target_count.index, target_count.values)
    # plt.show()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class:', target_count[ind_min_class])
    print('Majority class:', target_count[1 - ind_min_class])
    print('Proportion:', round(target_count[ind_min_class] / target_count[1 - ind_min_class], 2), ': 1')

    RANDOM_STATE = 42
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1 - ind_min_class]]}

    df_class_min = unbal[unbal['class'] == min_class]
    df_class_max = unbal[unbal['class'] != min_class]

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]

    smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
    y = unbal.pop('class').values
    X = unbal.values
    smote_x, smote_y = smote.fit_sample(X, y)
    new_data = []
    for i in range(len(smote_x)):
        new_data.append(np.append(smote_x[i], smote_y[i]))

    data = pd.DataFrame(data=new_data, columns=data[:0].columns)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1 - ind_min_class]]

    # plt.figure()
    # multiple_bar_chart(plt.gca(),
    #                         [target_count.index[ind_min_class], target_count.index[1-ind_min_class]],
    #                         values, 'Target', 'frequency', 'Class balance')
    # plt.show()
    return data


def get_class_balance_second_ds(data):
    unbal = copy(data)
    target_count = unbal['Cover_Type'].value_counts()
    # plt.figure()
    # plt.title('Class balance')
    # plt.bar(target_count.index, target_count.values)
    # plt.show()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    df_class_min = unbal[unbal['Cover_Type'] == min_class]

    aux = df_class_min
    for i in unbal['Cover_Type'].unique():
        if i != min_class:
            new_list = pd.DataFrame(unbal[unbal['Cover_Type'] == i])
            aux = pd.concat([aux, new_list.sample(len(df_class_min))])

    data = pd.DataFrame(data=aux, columns=data[:0].columns)
    return data


def erase_correlated_columns(data, threshold):
    corr_mtx = data.corr()
    keyId = 0
    value_id = 0
    keys = corr_mtx.keys()
    corr = []
    for i in corr_mtx.values:
        keyId = 0
        for j in i:
            if abs(j) > threshold and keyId != value_id:
                value = keys[value_id], keys[keyId]
                iValue = keys[keyId], keys[value_id]
                if not (iValue in corr):
                    corr.append(value)
            keyId += 1
        value_id += 1

    for v in corr:
        if v[1] in data.keys() and v[0] in data.keys():
            data = data.drop(columns=[v[1]])
    return data


def impute_missing_values_second_ds(original):
    cols_nr = original.select_dtypes(include='number')
    cols_sb = original.select_dtypes(include='category')

    imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
    imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
    if len(cols_nr.T) > 0:
        df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
    if len(cols_sb.T) > 0:
        df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

    if len(cols_nr.T) > 0 and len(cols_sb.T) > 0:
        data = df_nr.join(df_sb, how='right')
        data.describe(include='all')
    else:
        data = df_nr
    original = data
    aux = original.pop('Cover_Type').values
    data = Normalizer().fit_transform(original)
    data = pd.DataFrame(data, columns=original.columns)
    data.insert(data.shape[1], 'Cover_Type', aux)
    return data


def impute_missing_values(original):
    cols_nr = original.select_dtypes(include='number')
    cols_sb = original.select_dtypes(include='category')

    imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
    imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
    if len(cols_nr.T) > 0:
        df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
    if len(cols_sb.T) > 0:
        df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

    if len(cols_nr.T) > 0 and len(cols_sb.T) > 0:
        data = df_nr.join(df_sb, how='right')
        data.describe(include='all')
    else:
        data = df_nr
    original = data
    aux = original.pop('class').values
    data = Normalizer().fit_transform(original)
    data = pd.DataFrame(data, columns=original.columns)
    data.insert(data.shape[1], 'class', aux)
    return data


def dummify(df, cols_to_dummify):
    one_hot_encoder = OneHotEncoder(sparse=False)

    for var in cols_to_dummify:
        one_hot_encoder.fit(df[var].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([var])
        transformed_data = one_hot_encoder.transform(df[var].values.reshape(-1, 1))
        df = pd.concat((df, pd.DataFrame(transformed_data, columns=feature_names)), 1)
        df.pop(var)
    return df
