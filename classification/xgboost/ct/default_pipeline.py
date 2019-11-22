"""
Context: Cover Type Data Set
XGBoost results with apriori pre-processing.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import load_and_undersample_ct
from vis_functions import heatmap

# globals
LEARNING_RATE = [0.1, 0.2, 0.3]
MAX_DEPTH = [3, 6, 9]
NR_ESTIMATORS = [50, 100, 150]

METRICS = ["Accuracy", "Average Precision", "Average Recall"]
MAX_DEPTH_COLORS = plt.cm.rainbow(np.linspace(0, 1, len(MAX_DEPTH)))
PLOT_ACCURACY_ROW = 0
PLOT_PRECISION_ROW = 1
PLOT_RECALL_ROW = 2

#####
# load data
data = load_and_undersample_ct("../../../datasets/secondDataSet.csv")
target = data.pop("Cover_Type")
data = data.values

#####
# pre-processing
balanced_x, balanced_y = SMOTE(sampling_strategy='minority').fit_resample(data, target)
transformed_x = PCA(n_components=balanced_x.shape[1]).fit_transform(balanced_x)

# split into train/validation and test set
x_train_val, x_test, y_train_val, y_test = train_test_split(balanced_x, balanced_y, train_size=0.8, shuffle=True,
                                                            random_state=40, stratify=balanced_y)

# split into train and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.8, shuffle=True,
                                                  random_state=40, stratify=y_train_val)

#####
# parameter tuning
fig, axes = plt.subplots(3, 3)
bar_width = 0.3

# for each model combination
for learning_rate_i in range(len(LEARNING_RATE)):
    for max_depth_i in range(len(MAX_DEPTH)):
        for n_estimators_i in range(len(NR_ESTIMATORS)):
            learning_rate = LEARNING_RATE[learning_rate_i]
            max_depth = MAX_DEPTH[max_depth_i]
            n_estimators = NR_ESTIMATORS[n_estimators_i]
            print("evaluating validation set for model learning rate={}, max_depth={}, n_estimators={}"
                  .format(learning_rate, max_depth, n_estimators))

            # predict validation set
            clf = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
            clf.fit(x_train, y_train)
            y_predicted = clf.predict(x_val)

            # metrics
            accuracy = accuracy_score(y_val, y_predicted)
            precision = precision_score(y_val, y_predicted, average='macro')
            recall = recall_score(y_val, y_predicted, average='macro')

            # insert bar
            bar_x = n_estimators_i + (bar_width * max_depth_i)
            axes[PLOT_ACCURACY_ROW, learning_rate_i].bar(bar_x, accuracy, width=bar_width,
                                                         color=MAX_DEPTH_COLORS[max_depth_i])
            axes[PLOT_PRECISION_ROW, learning_rate_i].bar(bar_x, precision, width=bar_width,
                                                          color=MAX_DEPTH_COLORS[max_depth_i])
            axes[PLOT_RECALL_ROW, learning_rate_i].bar(bar_x, recall, width=bar_width,
                                                       color=MAX_DEPTH_COLORS[max_depth_i])

# insert title, labels, legend to the plots
xticks_pos = np.arange(len(NR_ESTIMATORS), dtype=np.float) + (bar_width * len(MAX_DEPTH) / 2)
for metric_i in range(len(METRICS)):
    for learning_rate_i in range(len(LEARNING_RATE)):
        # first row
        if metric_i == 0:
            axes[metric_i, learning_rate_i].set_title("Learning Rate={}".format(LEARNING_RATE[learning_rate_i]))

        axes[metric_i, learning_rate_i].set_xlabel("#estimators")
        axes[metric_i, learning_rate_i].set_ylabel(METRICS[metric_i])

        legend_list = []
        for i in range(len(MAX_DEPTH)):
            legend_list.append(mpatches.Patch(color=MAX_DEPTH_COLORS[i], label="max_depth={}".format(MAX_DEPTH[i])))

        axes[metric_i, learning_rate_i].legend(handles=legend_list)
        axes[metric_i, learning_rate_i].set_xticks(xticks_pos)
        axes[metric_i, learning_rate_i].set_xticklabels(np.array(NR_ESTIMATORS).astype(np.str))

#####
# fixed model evaluation
clf = XGBClassifier(learning_rate=0.2, max_depth=9, n_estimators=50)
clf.fit(x_train_val, y_train_val)
y_predicted = clf.predict(x_test)

# test metrics
acc_test = accuracy_score(y_test, y_predicted)
prec_test = precision_score(y_test, y_predicted, average='macro')
rec_test = recall_score(y_test, y_predicted, average='macro')

# print results
print("Evaluation on test set")
print("Accuracy={}, Precision={}, Recall={}".format(acc_test, prec_test, rec_test))

# confusion matrix
cm = confusion_matrix(y_test, y_predicted)
plt.figure()
heatmap(plt.gca(), cm, "Test Confusion Matrix", "predicted class", "true class")

plt.show()
