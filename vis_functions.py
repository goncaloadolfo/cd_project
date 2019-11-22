"""
Functions for visualization purposes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as _stats
import seaborn as sns


def choose_grid(nr):
    return nr // 4 + 1 if nr % 4 != 0 else nr // 4, 4 if nr >= 4 else nr


def line_chart(ax: plt.Axes, xseries: pd.Series, yseries: pd.Series, title: str, xlabel: str, ylabel: str,
               percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(xseries, yseries, "-o")


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                        percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox=True, shadow=True)


def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False,
              rotation=0):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=rotation, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')


def plot_matrix(ax: plt.Axes, mtx, labels, title: str, xlabel: str, ylabel: str, cmap='Pastel2'):
    im = ax.imshow(mtx, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(mtx.shape[1]))
    ax.set_yticks(np.arange(mtx.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_autoscale_on(False)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            ax.text(j, i, format(mtx[i, j], 'd'),
                    ha='center', va='center')
    ax.set_ylim(len(mtx) - .5, -.5)
    ax.get_figure().tight_layout()


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                       percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)


def draw_box_plot(ax, title, values, ylabel):
    # draw box plot for a column
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.boxplot(values)


def draw_simple_hist(ax, title, xlabel, values, bins):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counting")
    ax.hist(values, bins=bins)


def heatmap(ax, matrix, title, xlabel, ylabel):
    sns.heatmap(matrix, annot=True, ax=ax, cmap='Reds')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def scatter_plot(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, label: str, color,
                 alpha=1.0):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(xvalues, yvalues, color=color, label=label, alpha=alpha)


def __plot_variables__(data, plotter, dtype='number'):
    columns = data.select_dtypes(include=dtype).columns
    rows, cols = choose_grid(len(columns))
    # plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        plotter(axs[i, j], columns[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()


def variables_boxplot(data):
    def __plotter__(ax, column):
        ax.set_title('Boxplot for %s' % column)
        ax.boxplot(data[column].dropna().values)

    __plot_variables__(data, __plotter__)


def number_variables_histogram(data):
    def __plotter__(ax, column):
        ax.set_title('Histogram for %s' % column),
        ax.set_xlabel(column),
        ax.set_ylabel('probability'),
        ax.hist(data[column].dropna().values, 'auto')

    __plot_variables__(data, __plotter__)


def category_variables_histogram(data):
    def __plotter__(ax, column):
        counts = data[column].dropna().value_counts(normalize=True)
        bar_chart(ax, counts.index, counts.values, 'Histogram for %s' % column, column, 'probability')

    __plot_variables__(data, __plotter__, 'category')


def number_variables_histogram_with_trend(data):
    def __plotter__(ax, column):
        ax.set_title('Histogram with trend')
        ax.set_ylabel('probability')
        sns.distplot(data[column].dropna().values, norm_hist=True, ax=ax, axlabel=column)

    __plot_variables__(data, __plotter__)


def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussin
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)' % (mean, sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    # sigma, loc, scale = _stats.lognorm.fit(x_values)
    # distributions['LogNorm(%.1f,%.2f)' % (np.log(scale), sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)' % (1 / scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    # a, loc, scale = _stats.skewnorm.fit(x_values)
    # distributions['SkewNorm(%.2f)' % a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions


def number_variables_histogram_with_distributions(data):
    def __plotter__(ax, column):
        values = data[column].dropna().sort_values().values
        n, bins, patches = ax.hist(values, 20, density=True, edgecolor='grey')
        distributions = compute_known_distributions(values, bins)
        multiple_line_chart(ax, values, distributions, 'Best fit for %s' % column, column, 'probability')

    __plot_variables__(data, __plotter__)
