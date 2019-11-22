import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from classification.random_forests.random_forests_functions import *
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
    'accuracy': [],
    'recall': [],
    'precision': [],
    'roc-auc': []
}
for train_index, test_index in skf.split(X, y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = y[train_index], y[test_index]
    score = default_params_random_forest(score, trnX, trnY, tstX, tstY)

print_default_params_random_forest_scores(score, score_names)

n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
max_depths = [5, 10, 15, 20, 25, 50]
max_features = ['sqrt', 'log2']


def get_train_and_test_sets():
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        yield trnX, tstX, trnY, tstY


scores = parameterized_random_forests(get_train_and_test_sets, max_features, max_depths, n_estimators)
scores_mean = calculate_means(scores)

score_names = list(map(lambda s: s + '_mean', score_names))

print_and_plot_scores(scores_mean, score_names, max_features, max_depths, n_estimators)
