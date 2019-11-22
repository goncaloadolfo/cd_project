from typing import List, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest

from utils import remove_correlated_vars


def print_shape(data: pd.DataFrame):
    print('(#records, #variables) =', data.shape)
    print()


def print_variable_types(data: pd.DataFrame):
    print('variable types:')
    print('numeric - %d variables' % len(data.select_dtypes(include=['float64', 'int64']).columns))
    print('categoric - %d variables' % len(data.select_dtypes(include='object').columns))
    print()


def print_missing_values(data: pd.DataFrame):
    total_missing = 0
    for var in data:
        total_missing += data[var].isna().sum()
    print('Total missing values: %d' % total_missing)
    print()


def class_balance(data: pd.DataFrame, class_name: str, x_ticks: Iterable):
    target_count = data[class_name].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title('Class balance')
    ax.bar(target_count.index, target_count.values)
    ax.set_xticks(x_ticks)
    fig.tight_layout()
    plt.show()


def remove_corr_and_select_k_best(data: pd.DataFrame, k=4, class_name='class'):
    data = remove_correlated_vars(data, lambda x: x > .9 or x < -.9)

    y = data.pop(class_name)
    X = data.values

    selector = SelectKBest(k=k)
    selector.fit_transform(X, y)

    mask = selector.get_support()
    features_names = list(data.columns.values)
    new_features = []

    for boolean, feature in zip(mask, features_names):
        if boolean:
            new_features.append(feature)

    data = data[new_features]

    return data, X, y


def compare_distributions(datas: List[pd.DataFrame], column_idx, x_ticks, y_ticks, class_values):
    columns = datas[0].columns
    fig, axs = plt.subplots(1, len(datas), figsize=(4 * len(datas), 4), squeeze=False)

    for i in range(len(datas)):
        axs[0, i].set_title('class = %d' % class_values[i])
        axs[0, i].set_ylabel('probability')
        axs[0, i].set_autoscale_on(False)
        axs[0, i].set_xlim(x_ticks[0], x_ticks[-1])
        axs[0, i].set_ylim(y_ticks[0], y_ticks[-1])
        axs[0, i].set_xticks(x_ticks)
        axs[0, i].set_yticks(y_ticks)
        sns.distplot(datas[i][columns[column_idx]].values, norm_hist=True, ax=axs[0, i], axlabel=columns[column_idx])

    fig.tight_layout()
    plt.show()


def compare_binary_distributions(datas: List[pd.DataFrame], column_idx, y_ticks, class_values):
    columns = datas[0].columns
    fig, axs = plt.subplots(1, len(datas), figsize=(4 * len(datas), 4), squeeze=False)

    for i in range(len(datas)):
        column_count = datas[i][columns[column_idx]].value_counts()
        axs[0, i].set_title('class = %d' % class_values[i])
        axs[0, i].bar(column_count.index, column_count.values)
        axs[0, i].set_autoscale_on(False)
        axs[0, i].set_xlim(-.5, 1.5)
        axs[0, i].set_xticks([0, 1])
        axs[0, i].set_ylim(y_ticks[0], y_ticks[-1])
        axs[0, i].set_yticks(y_ticks)
        axs[0, i].set_xlabel(columns[column_idx])

    fig.tight_layout()
    plt.show()
