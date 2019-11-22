"""
XGBoost results for Parkinson Decease Data Set.
Tuning parameters: max_depth, learning_rate and n_estimators.
Pre-Processing: correlated variables removing, data balancing.
"""

# libs
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# own libs
from data_exploration.multi_analysis_functions import put_away_vars
from classification.xgboost.xgboost_functions import *


#####
# load data
data = pd.read_csv("../../datasets/pd_speech_features.csv")
y = data.pop('class').values

#####
# pre processing
vars_to_remove = put_away_vars(data.corr(), 0.9)
col_names_to_remove = data.columns[vars_to_remove]
data_array = data.drop(columns=col_names_to_remove).values

balanced_data, balanced_y = SMOTE(sampling_strategy='minority', k_neighbors=5).fit_resample(data_array, y)

x_train_val, x_test, y_train_val, y_test = train_test_split(balanced_data, balanced_y, train_size=0.75, shuffle=True)

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
