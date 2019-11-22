import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from classification.random_forests.random_forests_functions import *
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
    'accuracy': [],
    'recall': [],
    'precision': [],
    'roc-auc': []
}
score = default_params_random_forest(score, trnX, trnY, tstX, tstY, average_mode='micro')
print_default_params_random_forest_scores(score, score_names)

n_estimators = [10, 25, 50, 100, 200, 300]
max_depths = [25, 50, 75, 100]
max_features = ['sqrt', 'log2']


def get_train_and_test_sets():
    yield train_test_split(X, y, train_size=.7, stratify=y)


scores = parameterized_random_forests(get_train_and_test_sets, max_features, max_depths, n_estimators,
                                      average_mode='micro')
scores_mean = calculate_means(scores)

score_names = list(map(lambda s: s + '_mean', score_names))

print_and_plot_scores(scores_mean, score_names, max_features, max_depths, n_estimators)
