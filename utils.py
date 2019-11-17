'''
Auxiliar functions to other modules.
'''

# libs
import pandas as pd
import numpy as np


def print_return_variable(prefix, value):
    print(prefix + str(value))
    return value


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))


def undersampling_ct(ct_path):
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
