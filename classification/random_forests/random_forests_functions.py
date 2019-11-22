from functools import reduce
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import vis_functions as func


def calculate_means(scores: list):
    return list(map(lambda s: {
        'max_feature': s['max_feature'],
        'max_depth': s['max_depth'],
        'n_estimators': s['n_estimators'],
        'accuracy_mean': np.mean(s['accuracy']),
        'recall_mean': np.mean(s['recall']),
        'precision_mean': np.mean(s['precision']),
        'roc-auc_mean': np.mean(s['roc-auc'])
    }, scores))


def default_params_random_forest(score, trnX, trnY, tstX, tstY, average_mode='binary'):
    rf = RandomForestClassifier()
    rf.fit(trnX, trnY)
    prdY = rf.predict(tstX)
    score['accuracy'].append(metrics.accuracy_score(tstY, prdY))
    score['recall'].append(metrics.recall_score(tstY, prdY, average=average_mode))
    score['precision'].append(metrics.precision_score(tstY, prdY, average=average_mode))
    if average_mode == 'binary':
        score['roc-auc'].append(metrics.roc_auc_score(tstY, prdY))
    return score


def print_default_params_random_forest_scores(score, score_names: list):
    print('Default parameters random forest')
    print()
    for sn in score_names:
        print('%s:' % sn)
        print('Mean:', np.mean(score[sn]))
        print('Std: ', np.std(score[sn]))
        print('Max: ', max(score[sn]))
        print('Min: ', min(score[sn]))
        print()
    print('\n')


def parameterized_random_forests(get_train_and_test_sets, max_features: Iterable, max_depths: Iterable, n_estimators: Iterable,
                                 average_mode='binary'):
    scores = []
    for f in max_features:
        for d in max_depths:
            for n in n_estimators:
                s = {
                    'max_feature': f,
                    'max_depth': d,
                    'n_estimators': n,
                    'accuracy': [],
                    'recall': [],
                    'precision': [],
                    'roc-auc': []
                }
                for trnX, tstX, trnY, tstY in get_train_and_test_sets():
                    rf = RandomForestClassifier(n_estimators=n, max_features=f, max_depth=d)
                    rf.fit(trnX, trnY)
                    prdY = rf.predict(tstX)
                    s['accuracy'].append(metrics.accuracy_score(tstY, prdY))
                    s['recall'].append(metrics.recall_score(tstY, prdY, average=average_mode))
                    s['precision'].append(metrics.precision_score(tstY, prdY, average=average_mode))
                    if average_mode == 'binary':
                        s['roc-auc'].append(metrics.roc_auc_score(tstY, prdY))
                scores.append(s)
    return scores


def select_best_score(scores: list, score_name: str):
    return reduce(lambda acc, s: s if s[score_name] > acc[score_name] else acc, scores, {score_name: 0})


def print_and_plot_scores(scores, score_names: list, max_features: list, max_depths: Iterable, n_estimators: list):
    for score in score_names:
        print('Best %s params' % score)
        print(select_best_score(scores, score))
        all_values = {}
        for f in max_features:
            all_values[f] = {}
            for d in max_depths:
                all_values[f][d] = []
        for s in scores:
            all_values[s['max_feature']][s['max_depth']].append(s[score])

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        for i in range(len(max_features)):
            f = max_features[i]
            func.multiple_line_chart(axs[0, i], n_estimators, all_values[f], 'Random Forest with %s features' % f,
                                     'n_estimator', score, percentage=True)
        fig.tight_layout()
        plt.show()
        print()
