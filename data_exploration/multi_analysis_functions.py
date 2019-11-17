'''
Functions for data set multi analysis.
'''

# built-in
import pickle

# libs
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# globals
CORRELATION_BINS = 10


def get_next_vars_to_plot(plotted_vars, sorted_corr, original_corr_matrix, data):
    done = False
    var1, var2 = None, None
    corr_value = -5

    while not done:

        if len(sorted_corr) == 0:
            break

        next_value, sorted_corr = sorted_corr[0], sorted_corr[1:]
        rows, cols = np.where(original_corr_matrix == next_value)

        for index in range(len(rows)):
            row = rows[index]
            column = cols[index]

            if row != column and not (row, column) in plotted_vars and not (column, row) in plotted_vars:
                done = True
                plotted_vars.append((row, column))
                var1, var2 = data.columns[row], data.columns[column]
                corr_value = next_value

    return var1, var2, corr_value


def scatter_plots(data, target, correlation_matrix, criteria):
    flatted_array = correlation_matrix.ravel()

    # sort correlation values
    if criteria == -1:
        sorted_array = np.sort(flatted_array)

    elif criteria == 0:
        sorted_array = np.sort(np.abs(flatted_array))

    elif criteria == 1:
        sorted_array = np.sort(flatted_array)[::-1]

    else:
        raise ValueError("Sort criteria not accepted. Possible values: {-1, 0, 1}")

    # create subplots
    fig = plt.figure("Correlation analysis, criteria=" + str(criteria), figsize=(16, 8))
    axes = fig.subplots(2, 4, squeeze=False)
    plotted_vars = []

    for col in range(4):
        for row in range(2):
            var1, var2, corr_value = get_next_vars_to_plot(plotted_vars, sorted_array, correlation_matrix, data)

            # if there are more variables to see
            if corr_value != -5:
                axes[row, col].set_title("corr=" + str(format(corr_value, '.4f')))
                axes[row, col].set_xlabel(var1)
                axes[row, col].set_ylabel(var2)
                axes[row, col].plot(data[var1][target == 0], data[var2][target == 0], '^')
                axes[row, col].plot(data[var1][target == 1], data[var2][target == 1], 's')

            else:
                break

        if len(sorted_array) == 0:
            break

    plt.tight_layout()


def draw_heatmap(matrix, xticklabels, yticklabels, annot, cmap, title, xlabel, ylabel, axes):
    sns.heatmap(matrix, xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, cmap=cmap, ax=axes)
    axes.set_title(title)
    axes.set_yticks(range(len(yticklabels)))
    axes.set_xticks(range(len(xticklabels)))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)


def correlation_histogram(title, corr_matrix, axes):
    # delete diagonal values
    np.fill_diagonal(corr_matrix, -5)
    corr_values = corr_matrix[np.where(corr_matrix != -5)]
    nr_corrs = len(corr_values)

    # draw histogram
    edge_bins = np.arange(-1.0, 1.00001, step=2 / CORRELATION_BINS)
    axes.set_title(title)
    axes.set_xlabel("correlation")
    axes.set_ylabel("percentage(%)")
    axes.set_xticks(edge_bins)
    n, bins, patches = axes.hist(corr_values, weights=(np.ones(nr_corrs) / nr_corrs) * 100,
                                bins=CORRELATION_BINS, range=(-1.0, 1.0))
    axes.grid()

    print("######")
    print(title)
    for i in range(len(n)):
        print(str(format(bins[i], '.2f')) + "-" + str(format(bins[i + 1], '.2f')) + ": " + str(format(n[i], '.2f')))


def put_away_vars(corr_matrix, thr):
    return_vars = []
    # apply predicate
    rows, cols = np.where(np.logical_or(corr_matrix > thr, corr_matrix < -thr))

    # for each result
    for i in range(len(rows)):
        # get variable indexes
        var1 = rows[i]
        var2 = cols[i]

        # if none of them is putted away already
        if not(var1 in return_vars or var2 in return_vars) and var1 != var2:
            return_vars.append(var1)

    return return_vars


def corr_thr(corr_matrix, thresholds):
    vars_thr = []

    for thr in thresholds:
        n = corr_matrix.shape[0] - len(put_away_vars(corr_matrix, thr))
        vars_thr.append(n)

    # draw plot with results
    plt.figure("Correlation threshold")
    plt.title("N variables/threshold")
    plt.xlabel("correlation thr")
    plt.ylabel("dimensionality")
    plt.plot(thresholds, vars_thr, marker='o', linestyle='--')
    plt.grid()


def correlation_analysis(data, target, group_title, hist_axes, heatmap_axes, draw_scatter_plots=False):
    corr_matrix = data.corr().values

    # correlation heatmap
    nr_variables = len(data.columns)
    ticks = np.linspace(0, nr_variables - 1, 5, dtype=np.int) if nr_variables > 15 else np.arange(nr_variables)
    draw_heatmap(corr_matrix, ticks, ticks, False, 'Reds', group_title + ' correlation matrix', 'variable index',
                 'variable index', heatmap_axes)

    # correlation histogram
    correlation_histogram(group_title + " correlation histogram", corr_matrix, hist_axes)

    # scatter plots of most/lowest correlated variables
    if draw_scatter_plots:
        scatter_plots(data, target, corr_matrix, -1)
        scatter_plots(data, target, corr_matrix, 1)
        scatter_plots(data, target, corr_matrix, 0)


def multi_analysis(data, target, attr_groups):
    # analyse correlation in each atr group
    attr_group_names = list(attr_groups.keys())
    figures_needed = int(len(attr_group_names) / 3) if len(attr_group_names) % 3 == 0 else \
        int(len(attr_group_names) / 3) + 1

    for figure in range(figures_needed):
        # create a figure for each 4 atr groups
        fig = plt.figure("Attrs group correlation " + str(figure), figsize=(16, 8))
        axes = fig.subplots(2, 3, squeeze=False)

        last_index = figure * 3 + 3
        last_index = last_index if last_index < len(attr_group_names) else len(attr_group_names)
        for group_name_index in range(figure * 3, last_index):
            # atr group description
            group_name = attr_group_names[group_name_index]

            # atrs index ranges
            index_range = attr_groups[group_name]

            # get subset data
            data_subset = data.iloc[:, index_range[0]: index_range[1] + 1]

            # correlation of that subset
            correlation_analysis(data_subset, target, group_name, axes[0, group_name_index % 3],
                                 axes[1, group_name_index % 3])

        plt.tight_layout()

    # discover a nice correlation thr
    corr_thr(data.corr().values, np.arange(0.5, 1.01, 0.05))

    # remove correlated vars and check inter-group correlation hist and heatmap
    vars_to_remove = put_away_vars(data.corr(), 0.85)
    columns_to_remove = data.columns[vars_to_remove]
    new_df = data.drop(columns=columns_to_remove)

    fig = plt.figure('After removing correlated vars')
    axes = fig.subplots(1, 2, squeeze=False)
    correlation_analysis(new_df, target, 'Global', axes[0, 0], axes[0, 1])
    fig.tight_layout()


def save_datasets(data, correlation_thresholds, target):
    # information needed
    corr_matrix = data.corr()
    columns = data.columns
    datasets = {}

    for thr in correlation_thresholds:
        # remove correlated variables (|corr| > thr)
        vars_to_remove = put_away_vars(corr_matrix, thr)
        column_names_to_remove = columns[vars_to_remove]
        new_df = data.drop(columns=column_names_to_remove)

        # save dataset without that variables
        datasets[thr] = new_df.values

    datasets['target'] = target
    # write datasets into a pickle file
    with open("parkinson_datasets.p", "wb") as file:
        pickle.dump(datasets, file, protocol=pickle.HIGHEST_PROTOCOL)
