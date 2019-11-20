"""
XGBoost results for Parkinson Decease Data Set.
Tuning parameters: max_depth, learning_rate and n_estimators.
"""

# libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def draw_lr_results(ax: plt.Axes, title, lr_results: np.ndarray, n_estimators_labels: list,
                    max_depth_labels: list) -> None:
    # title and labels
    ax.set_title(title)
    ax.set_xlabel("#estimators")
    ax.set_ylabel("recall")

    # each max depth results with different color
    colors = plt.cm.rainbow(np.linspace(0, 1, len(max_depth_labels)))

    for max_depth_i in range(len(max_depth_labels)):
        max_depth = max_depth_labels[max_depth_i]
        recalls = lr_results[max_depth_i]
        ax.plot(n_estimators_labels, recalls, "-o", label="max_depth=%d" % max_depth, c=colors[max_depth_i])

    ax.legend()


#####
# load data
data = pd.read_csv("../../datasets/pd_speech_features.csv")
y = data.pop('class').values
data = data.values

#####
# parameter tuning
max_depth = [3, 6, 9]
learning_rate = [0.05, 0.1, 0.2]
n_estimators = [50, 100, 150, 200, 250, 300]

results_shape = (len(learning_rate), len(max_depth), len(n_estimators))
recalls = np.zeros(results_shape, dtype=np.float)

print("##### Parameter tuning")
for curr_max_depth_i in range(len(max_depth)):
    curr_max_depth = max_depth[curr_max_depth_i]

    for curr_learning_rate_i in range(len(learning_rate)):
        curr_learning_rate = learning_rate[curr_learning_rate_i]

        for curr_n_estimators_i in range(len(n_estimators)):

            curr_n_estimators = n_estimators[curr_n_estimators_i]
            print("current model: "
                  "max_depth=" + str(curr_max_depth) +
                  " learning_rate=" + str(curr_learning_rate) +
                  " n_estimators=" + str(curr_n_estimators))

            xgboost = XGBClassifier(max_depth=curr_max_depth, learning_rate=curr_learning_rate,
                                    n_estimators=curr_n_estimators)
            strat_kfold_obj = StratifiedKFold(n_splits=5, shuffle=True)

            recall_per_fold = []
            for fold in strat_kfold_obj.split(data, y):
                train_indexes = fold[0]
                test_indexes = fold[1]

                x_train, y_train = data[train_indexes], y[train_indexes]
                x_test, y_test = data[test_indexes], y[test_indexes]

                xgboost.fit(x_train, y_train)
                y_predicted = xgboost.predict(x_test)
                recall_per_fold.append(recall_score(y_test, y_predicted))

            model_average_recall = np.mean(recall_per_fold)
            recalls[curr_learning_rate_i, curr_max_depth_i, curr_n_estimators_i] = model_average_recall

fig, axes = plt.subplots(1, len(learning_rate))
for i in range(len(learning_rate)):
    draw_lr_results(axes[i], "learning rate=" + str(learning_rate[i]), recalls[i], n_estimators, max_depth)

plt.tight_layout()
plt.show()
