'''
Functions for visualization purposes.
'''

# libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def choose_grid(nr):
    return 1 if nr <= 4 else 2, nr if nr < 4 else 4


def line_chart(ax: plt.Axes, xseries: pd.Series, yseries: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(xseries, yseries, "-o")


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False,
                        legend=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)

    if legend:
        ax.legend(legend, loc='best', fancybox = True, shadow = True)


def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False, rotation=0):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=rotation, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):
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
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox = True, shadow = True)


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


def heatmap(ax, matrix, title, xlabel, ylabel, annot=True):
    sns.heatmap(matrix, annot=False, ax=ax, cmap='Reds')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def scatter_plot(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, label: str, color,
                 alpha=1.0):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(xvalues, yvalues, color=color, label=label, alpha=alpha)
