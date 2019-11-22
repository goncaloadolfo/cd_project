"""
Common functions for XGBoost classification.
"""

# libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# own libs
from vis_functions import line_chart, heatmap

# globals
MAX_DEPTH_COL = "max_depth"
LEARNING_RATE_COL = "learning_rate"
N_ESTIMATORS_COL = "n_estimators"
AVERAGE_PRECISION_COL = "average_precision"
AVERAGE_RECALL_COL = "average_recall"
AVERAGE_ACCURACY_COL = "average_accuracy"


def cross_val_tuning(x_train_val: np.ndarray, y_train_val: np.ndarray,
                     max_depth: list, learning_rate: list, n_estimators: list) -> pd.DataFrame:
    search_results = []

    print("##### Parameter tuning")
    # for each model combination
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

                precision_per_fold = []
                recall_per_fold = []
                accuracy_per_fold = []

                # stratified cross validation
                for fold in strat_kfold_obj.split(x_train_val, y_train_val):
                    train_indexes = fold[0]
                    test_indexes = fold[1]

                    x_train, y_train = x_train_val[train_indexes], y_train_val[train_indexes]
                    x_val, y_val = x_train_val[test_indexes], y_train_val[test_indexes]

                    xgboost.fit(x_train, y_train)
                    y_predicted = xgboost.predict(x_val)

                    precision_per_fold.append(precision_score(y_val, y_predicted))
                    recall_per_fold.append(recall_score(y_val, y_predicted))
                    accuracy_per_fold.append(accuracy_score(y_val, y_predicted))

                # mean precision, recall and accuracy per fold
                model_average_precision = np.mean(precision_per_fold)
                model_average_recall = np.mean(recall_per_fold)
                model_average_accuracy = np.mean(accuracy_per_fold)

                search_results.append([curr_n_estimators, curr_learning_rate, curr_max_depth, model_average_precision,
                                       model_average_recall, model_average_accuracy])

    results_pd = pd.DataFrame(search_results, columns=[N_ESTIMATORS_COL, LEARNING_RATE_COL, MAX_DEPTH_COL,
                                                       AVERAGE_PRECISION_COL, AVERAGE_RECALL_COL, AVERAGE_ACCURACY_COL])
    print(results_pd)
    return results_pd


def split_tuning(x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, max_depth: list,
                 learning_rate: list, n_estimators: list) -> pd.DataFrame:
    search_results = []

    print("##### Parameter tuning")
    # for each model combination
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

                xgboost.fit(x_train, y_train)
                y_predicted = xgboost.predict(x_val)

                # mean precision, recall and accuracy per fold
                precision = precision_score(y_val, y_predicted, average='weighted')
                recall = recall_score(y_val, y_predicted, average='weighted')
                accuracy = accuracy_score(y_val, y_predicted)

                search_results.append([curr_n_estimators, curr_learning_rate, curr_max_depth, precision,
                                       recall, accuracy])

    results_pd = pd.DataFrame(search_results, columns=[N_ESTIMATORS_COL, LEARNING_RATE_COL, MAX_DEPTH_COL,
                                                       AVERAGE_PRECISION_COL, AVERAGE_RECALL_COL, AVERAGE_ACCURACY_COL])
    print(results_pd)
    return results_pd


def tuning_results_plot(results_pd: pd.DataFrame, max_depth: list, learning_rate: list, n_estimators: list) -> None:
    metrics_list = [AVERAGE_RECALL_COL, AVERAGE_PRECISION_COL, AVERAGE_ACCURACY_COL]
    bar_colors = plt.cm.rainbow(np.linspace(0, 1, len(max_depth)))
    fig, axes = plt.subplots(3, 3)

    bar_width = 0.3
    # results fig
    for metrics_i in range(len(metrics_list)):
        for lr_i in range(len(learning_rate)):
            bars_xpos = np.arange(len(max_depth), dtype=np.float)
            for md_i in range(len(max_depth)):
                metric = metrics_list[metrics_i]
                lr = learning_rate[lr_i]
                md = max_depth[md_i]

                combo_results = results_pd[np.logical_and(
                    results_pd[LEARNING_RATE_COL] == lr,
                    results_pd[MAX_DEPTH_COL] == md)]
                sorted_results = combo_results.sort_values(N_ESTIMATORS_COL)[metric].values

                axes[metrics_i, lr_i].set_title("Learning Rate={}".format(learning_rate[lr_i]))
                axes[metrics_i, lr_i].set_xlabel("#estimators")
                axes[metrics_i, lr_i].set_ylabel(metric)
                axes[metrics_i, lr_i].bar(bars_xpos, sorted_results, bar_width, label="max_depth={}".format(md),
                                          color=bar_colors[md_i])
                bars_xpos += bar_width
            axes[metrics_i, lr_i].legend()
            axes[metrics_i, lr_i].set_xticks(bars_xpos)
            axes[metrics_i, lr_i].set_xticklabels(np.array(n_estimators).astype(np.str))


def fixed_model_evaluation(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                           chosen_max_depth: int, chosen_lr: float, chosen_n_estimators: int) -> None:
    xgboost = XGBClassifier(max_depth=chosen_max_depth, learning_rate=chosen_lr, n_estimators=chosen_n_estimators)
    xgboost.fit(x_train, y_train)
    train_prediction = xgboost.predict(x_train)
    y_predicted = xgboost.predict(x_test)

    # support, precision, recall, accuracy, ...
    print("##### Fixed model evaluation")
    print("max_depth={}, learning_rate={}, n_estimators={}".format(chosen_max_depth, chosen_lr, chosen_n_estimators))
    print("train evaluation: acc={}, precision={}, recall={}".format(
        accuracy_score(y_train, train_prediction),
        precision_score(y_train, train_prediction),
        recall_score(y_train, train_prediction)))
    print("test evaluation: acc={}, precision={}, recall={}".format(
        accuracy_score(y_test, y_predicted),
        precision_score(y_test, y_predicted),
        recall_score(y_test, y_predicted)))

    # confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predicted)
    plt.figure()
    heatmap(plt.gca(), conf_matrix, "Confusion Matrix", "true class", "predicted class")

    # precision-recall curve and auc
    proba = xgboost.predict_proba(x_test)
    precisions, recalls, _ = precision_recall_curve(y_test, proba[:, 1])
    plt.figure()
    line_chart(plt.gca(), recalls, precisions, "Precision-Recall Curve", "recall", "precision")
