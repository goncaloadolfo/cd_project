"""
Obtaining association rules for PD dataset.
"""

from sklearn.feature_selection import f_classif

from pattern_mining.pattern_mining_functions import *
from utils import load_pd, print_return_variable, remove_correlated_vars

data, X, y = load_pd("../datasets/pd_speech_features.csv")

# remove high correlated variables
new_df = remove_correlated_vars(data, lambda x: x > .8 or x < -.8)

# pattern mining parameters
print("\n### Pattern mining parameters")
k_features = print_return_variable("Number of features to select: ", 35)
selection_measure = print_return_variable("Feature selection function: ", f_classif)
discretize_function = print_return_variable("Discretize function: ", pd.cut)
bins = print_return_variable("Number of bins: ", 7)
disc_needless_cols = print_return_variable("Variables that doesnt need to be discretize: ", ['gender', 'class'])
min_supp = print_return_variable("Min support: ", 0.6)
fp_mining_args = [min_supp]
min_conf = print_return_variable("Min confidence: ", 0.9)
min_ant_items = print_return_variable("Min of items in antecedents itemset: ", 2)

# get results
selected_features, dummified_df, frequent_patterns, _, rules = pm_system(new_df, y, k_features, selection_measure,
                                                                         discretize_function, bins,
                                                                         disc_needless_cols, fp_mining_args,
                                                                         min_conf, min_ant_items)

# results visualization
print("\n### Selected features")
print("Selected features: ", selected_features)
print("\n### Frequent patterns found (some)")
pretty_print_fsets(frequent_patterns, False, 5)
print("\n### Association rules found (some)")
pretty_print_rules(rules, ['confidence'], False, 10)

fsets_per_supp(dummified_df, np.arange(0.4, 1.001, 0.1))
