"""
Auxiliar functions to other modules.
"""

import numpy as np
import pandas as pd


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
