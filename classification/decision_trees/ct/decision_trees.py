import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from classification.decision_trees.decision_trees_functions import *
from utils import load_and_undersample_ct

data: pd.DataFrame
y: np.ndarray
X: np.ndarray

data = load_and_undersample_ct('../../../datasets/secondDataSet.csv')
y = data.pop('Cover_Type')
X = data.values

# SMOTE balancing
RANDOM_STATE = 42
smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
X, y = smote.fit_sample(X, y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=.7, stratify=y)

score_names = ['accuracy']

score = {
    'tree_depths': [],
    'tree_leafs': [],
    'accuracy': [],
    'recall': [],
    'precision': [],
    'roc-auc': []
}
score = default_params_tree(score, trnX, trnY, tstX, tstY, average_mode='micro')
print_default_params_tree_scores(score, score_names)

# #              [25429,12715,5086,2543,1272, 509,  255,   128,   51,    26,     13,     6,      3,       2,      1]
# min_samples_leaf = [.5, .25, .1, .05, .025, .01, .005, .0025, .001, .0005, .00025, .0001, .00005, .000025, .00001]
min_samples_leaf = [.0005, .00025, .0001, .00005, .000025, .00001]
max_depths = range(22, 28)
criteria = ['entropy', 'gini']


def get_train_and_test_sets():
    yield train_test_split(X, y, train_size=.7, stratify=y)


scores = parameterized_tree(get_train_and_test_sets, criteria, max_depths, min_samples_leaf, average_mode='micro')
scores_mean = calculate_means(scores)

score_names = list(map(lambda s: s + '_mean', score_names))

print_and_plot_scores(scores_mean, score_names, criteria, max_depths, min_samples_leaf)
