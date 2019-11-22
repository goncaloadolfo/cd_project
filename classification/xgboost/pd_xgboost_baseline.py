"""
Script to have a baseline of XGBoost performance.
Only tuning is applied.
"""

# libs
from sklearn.model_selection import train_test_split

# own libs
from classification.xgboost.xgboost_functions import *


#####
# load data
data = pd.read_csv("../../datasets/pd_speech_features.csv")
y = data.pop('class').values

x_train_val, x_test, y_train_val, y_test = train_test_split(data.values, y, train_size=0.75, shuffle=True)

#####
# parameter tuning
max_depth = [3, 6, 9]
learning_rate = [0.1, 0.2, 0.3]
n_estimators = [50, 100, 150]
results_pd = cross_val_tuning(x_train_val, y_train_val, max_depth, learning_rate, n_estimators)
tuning_results_plot(results_pd, max_depth, learning_rate, n_estimators)

#####
# fixed model evaluation
chosen_max_depth = 3
chosen_lr = 0.3
chosen_n_estimators = 50
fixed_model_evaluation(x_train_val, y_train_val, x_test, y_test, chosen_max_depth, chosen_lr, chosen_n_estimators)

plt.tight_layout()
plt.show()
