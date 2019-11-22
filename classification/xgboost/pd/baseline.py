"""
Context: Parkinson Decease Data Set
XGBoost results without pre-processing and parameter tuning.
"""

# libs
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

# own libs
from vis_functions import line_chart

# globals
NR_FOLDS = 5


# load data
data = pd.read_csv("../../../datasets/pd_speech_features.csv").values
target = data[:, -1]

# classifier and kfold obj
xgboost = XGBClassifier()
strat_obj = StratifiedKFold(n_splits=NR_FOLDS, shuffle=True)
acc_per_fold = []
prec_per_fold = []
rec_per_fold = []

# cross validation
fold_index = 0
for fold in strat_obj.split(data, target):
    train_indexes = fold[0]
    test_indexes = fold[1]

    train_x, train_y = data[train_indexes], target[train_indexes]
    test_x, test_y = data[test_indexes], target[test_indexes]

    xgboost.fit(train_x, train_y)
    print("evaluating fold {}".format(fold_index))
    y_predicted = xgboost.predict(test_x)

    # metrics
    acc_per_fold.append(accuracy_score(test_y, y_predicted))
    prec_per_fold.append(precision_score(test_y, y_predicted))
    rec_per_fold.append(recall_score(test_y, y_predicted))

    fold_index += 1

# results plot
fig, axes = plt.subplots(1, 3)
line_chart(axes[0], list(range(NR_FOLDS)), acc_per_fold, "Accuracy per fold", "fold", "accuracy")
line_chart(axes[1], list(range(NR_FOLDS)), rec_per_fold, "Recall per fold", "fold", "recall")
line_chart(axes[2], list(range(NR_FOLDS)), prec_per_fold, "Precision per fold", "fold", "precision")
plt.show()
