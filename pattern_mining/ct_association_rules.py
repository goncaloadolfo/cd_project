'''
Association rules for Cover Type data set.
'''

# own libs
from utils import undersampling_ct, print_return_variable
from pattern_mining.pattern_mining_functions import *
from mlxtend.frequent_patterns import fpgrowth


# load data set and under sample it
ct_data = undersampling_ct("../datasets/secondDataSet.csv")
variables_names = ct_data.columns

# pattern mining parameters
print("\n### Pattern mining parameters")

discretize_function = print_return_variable("Discretize function: ", pd.cut)
bins = print_return_variable("Number of bins: ", 4)
disc_needless_cols = print_return_variable("Variables that doesnt need to be discretize: ", variables_names[10:])
min_supp = print_return_variable("Min support: ", 0.6)
fp_mining_args = [min_supp]
min_conf = print_return_variable("Min confidence: ", 0.9)
min_ant_items = print_return_variable("Min of items in antecedents itemset: ", 2)

# get results
ct_subset = ct_data[ct_data['Cover_Type'] == 2].iloc[:10]
dummy_df = reduce_discretize_dummify_df(ct_subset, variables_names, disc_needless_cols, discretize_function, bins)
fpgrowth(dummy_df, min_support=0.9)
