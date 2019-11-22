"""
Association rules for Cover Type data set.
"""

# libs
from sklearn.feature_selection import f_classif

from pattern_mining.pattern_mining_functions import *


#####
# load data
ct_data = undersampling_ct("../datasets/secondDataSet.csv")
y = ct_data.pop('Cover_Type').values
columns = ct_data.columns.values

#####
# pattern mining parameters
print("\n### Pattern mining parameters")

k_features = print_return_variable("Number of features to select: ", 54)
selection_measure = print_return_variable("Feature selection function: ", f_classif)
discretize_function = print_return_variable("Discretize function: ", pd.cut)
bins = print_return_variable("Number of bins: ", 7)
disc_needless_cols = print_return_variable("Variables that doesnt need to be discretize: ", columns[10:])
binary_cols = print_return_variable("Binary cols: ", columns[10:])
min_supp = print_return_variable("Min support: ", 0.2)
fp_mining_args = [min_supp]
min_conf = print_return_variable("Min confidence: ", 0.6)
min_ant_items = print_return_variable("Min of items in antecedents itemset: ", 1)

#####
# AR results
selected_features, dummified_df, frequent_patterns, _, rules = pm_system(ct_data, y, k_features, selection_measure,
                                                                         discretize_function, bins,
                                                                         disc_needless_cols, binary_cols,
                                                                         fp_mining_args, min_conf, min_ant_items)

print("\n### Selected features")
print("Selected features: ", selected_features)
print("\n### Frequent patterns found (some)")
pretty_print_fsets(frequent_patterns, False, 5)
print("\n### Association rules found (some)")
pretty_print_rules(rules, ['confidence'], False, 20)

fsets_per_supp(dummified_df, np.arange(0.1, 1.001, 0.1))
