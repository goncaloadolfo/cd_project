"""
Script to have a baseline of XGBoost performance.
Only tuning is applied.
"""

# libs
from sklearn.model_selection import train_test_split

# own libs
from classification.xgboost.xgboost_functions import *
from utils import undersampling_ct


#####
# load data
data = undersampling_ct("../../datasets/secondDataSet.csv")
y = data.pop('Cover_Type').values

x_train_val, x_test, y_train_val, y_test = train_test_split(data.values, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.8, shuffle=True)

#####
# parameter tuning
max_depth = [3, 6, 9]
learning_rate = [0.1, 0.2, 0.3]
n_estimators = [50, 100, 150]
results_pd = split_tuning(x_train, x_val, y_train, y_val, max_depth, learning_rate, n_estimators)
tuning_results_plot(results_pd, max_depth, learning_rate, n_estimators)

#####
# fixed model evaluation
chosen_max_depth = 3
chosen_lr = 0.3
chosen_n_estimators = 50

xgboost = XGBClassifier(max_depth=chosen_max_depth, learning_rate=chosen_lr, n_estimators=chosen_n_estimators)
xgboost.fit(x_train, y_train)
train_prediction = xgboost.predict(x_train)
y_predicted = xgboost.predict(x_test)

print("##### Fixed model evaluation")
print("max_depth={}, learning_rate={}, n_estimators={}".format(chosen_max_depth, chosen_lr, chosen_n_estimators))
print("train evaluation: acc={}, precision={}, recall={}".format(
    accuracy_score(y_train, train_prediction),
    precision_score(y_train, train_prediction, average='macro'),
    recall_score(y_train, train_prediction, average='macro')))
print("test evaluation: acc={}, precision={}, recall={}".format(
    accuracy_score(y_test, y_predicted),
    precision_score(y_test, y_predicted, average='macro'),
    recall_score(y_test, y_predicted, average='macro')))

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_predicted)
plt.figure()
heatmap(plt.gca(), conf_matrix, "Confusion Matrix", "true class", "predicted class")

plt.tight_layout()
plt.show()
