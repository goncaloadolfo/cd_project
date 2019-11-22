import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from classification.decision_trees.decision_trees_functions import *
from utils import load_pd

data: pd.DataFrame
y: np.ndarray
X: np.ndarray

data, X, y = load_pd('../../../datasets/pd_speech_features.csv', merge_observations=True)

# SMOTE balancing
RANDOM_STATE = 42
smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
X, y = smote.fit_sample(X, y)

skf = StratifiedKFold(5)

score_names = ['accuracy', 'recall', 'precision', 'roc-auc']
score = {
    'tree_depths': [],
    'tree_leafs': [],
    'accuracy': [],
    'recall': [],
    'precision': [],
    'roc-auc': []
}
for train_index, test_index in skf.split(X, y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = y[train_index], y[test_index]
    score = default_params_tree(score, trnX, trnY, tstX, tstY)

print_default_params_tree_scores(score, score_names)

#                  [126, 63, 26,  13,    7,   3,    2,     1]
min_samples_leaf = [.5, .25, .1, .05, .025, .01, .005, .0025]
# min_samples_leaf = [.5, .35, .3, .25, .2, .1, .05, .025]  # closing in on recall peak
# min_samples_leaf = [.5, .3, .27, .25, .23, .2, .1, .05]  # closing in even more on recall peak
# min_samples_leaf = [.5, .3, .27, .26, .25, .24, .23, .22, .21, .2, .1, .01]  # closing in even more on recall peak
max_depths = range(4, 12)
criteria = ['entropy', 'gini']


def get_train_and_test_sets():
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        yield trnX, tstX, trnY, tstY


scores = parameterized_tree(get_train_and_test_sets, criteria, max_depths, min_samples_leaf)
scores_mean = calculate_means(scores)

score_names = list(map(lambda s: s + '_mean', score_names))

print_and_plot_scores(scores_mean, score_names, criteria, max_depths, min_samples_leaf)
