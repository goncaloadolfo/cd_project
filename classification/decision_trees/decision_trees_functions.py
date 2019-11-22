from functools import reduce
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import vis_functions as func


def calculate_means(scores: list):
    return list(map(lambda s: {
        'criteria': s['criteria'],
        'max_depth': s['max_depth'],
        'min_samples_leaf': s['min_samples_leaf'],
        'tree_depths': s['tree_depths'],
        'tree_leafs': s['tree_leafs'],
        'accuracy_mean': np.mean(s['accuracy']),
        'recall_mean': np.mean(s['recall']),
        'precision_mean': np.mean(s['precision']),
        'roc-auc_mean': np.mean(s['roc-auc'])
    }, scores))


def default_params_tree(score, trnX, trnY, tstX, tstY, average_mode='binary'):
    tree = DecisionTreeClassifier()
    tree.fit(trnX, trnY)
    prdY = tree.predict(tstX)
    score['tree_depths'].append(tree.get_depth())
    score['tree_leafs'].append(tree.get_n_leaves())
    score['accuracy'].append(metrics.accuracy_score(tstY, prdY))
    score['recall'].append(metrics.recall_score(tstY, prdY, average=average_mode))
    score['precision'].append(metrics.precision_score(tstY, prdY, average=average_mode))
    if average_mode == 'binary':
        score['roc-auc'].append(metrics.roc_auc_score(tstY, prdY))
    return score


def print_default_params_tree_scores(score, score_names: list):
    print('Default parameters tree')
    print('Tree depths:', score['tree_depths'])
    print('Tree leafs:', score['tree_leafs'])
    print()
    for sn in score_names:
        print('%s:' % sn)
        print('Mean:', np.mean(score[sn]))
        print('Std: ', np.std(score[sn]))
        print('Max: ', max(score[sn]))
        print('Min: ', min(score[sn]))
        print()
    print('\n')


def parameterized_tree(get_train_and_test_sets, criteria: Iterable, max_depths: Iterable, min_samples_leaf: Iterable,
                       average_mode='binary'):
    scores = []
    for f in criteria:
        for d in max_depths:
            for n in min_samples_leaf:
                s = {
                    'criteria': f,
                    'max_depth': d,
                    'min_samples_leaf': n,
                    'tree_depths': [],
                    'tree_leafs': [],
                    'accuracy': [],
                    'recall': [],
                    'precision': [],
                    'roc-auc': []
                }
                for trnX, tstX, trnY, tstY in get_train_and_test_sets():
                    tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                    tree.fit(trnX, trnY)
                    prdY = tree.predict(tstX)
                    s['tree_depths'].append(tree.get_depth())
                    s['tree_leafs'].append(tree.get_n_leaves())
                    s['accuracy'].append(metrics.accuracy_score(tstY, prdY))
                    s['recall'].append(metrics.recall_score(tstY, prdY, average=average_mode))
                    s['precision'].append(metrics.precision_score(tstY, prdY, average=average_mode))
                    if average_mode == 'binary':
                        s['roc-auc'].append(metrics.roc_auc_score(tstY, prdY))
                scores.append(s)
    return scores


def select_best_score(scores: list, score_name: str):
    return reduce(lambda acc, s: s if s[score_name] > acc[score_name] else acc, scores, {score_name: 0})


def print_and_plot_scores(scores, score_names: list, criteria: list, max_depths: Iterable, min_samples_leaf: list):
    for score in score_names:
        print('Best %s params' % score)
        print(select_best_score(scores, score))
        all_values = {}
        for f in criteria:
            all_values[f] = {}
            for d in max_depths:
                all_values[f][d] = []
        for s in scores:
            all_values[s['criteria']][s['max_depth']].append(s[score])

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        for i in range(len(criteria)):
            f = criteria[i]
            func.multiple_line_chart(axs[0, i], min_samples_leaf, all_values[f], 'Decision Trees with %s criteria' % f,
                                     'min_samples_leaf', score, percentage=True)
        fig.tight_layout()
        plt.show()
        print()
