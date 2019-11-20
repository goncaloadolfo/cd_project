'''
Pattern Mining functions.
'''

# built-in
import operator

# libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from mlxtend.frequent_patterns import association_rules, fpgrowth


#######
# Transactional data encoding functions
def feature_selection_dc(data: np.ndarray, target: np.ndarray, column_names: pd.Index, disc_cap_function,
                         n: int) -> tuple:
    # reduce data dimensionality based on features discriminative measure
    seleckkbest_obj = SelectKBest(disc_cap_function, n)
    reduced_data = seleckkbest_obj.fit_transform(data, target)

    # get column names of the most discriminative features
    feature_scores = seleckkbest_obj.scores_
    feature_score_tuples = [(column_names[i], feature_scores[i]) for i in range(len(column_names))]
    sorted_features = sorted(feature_score_tuples, key=operator.itemgetter(1), reverse=True)

    return reduced_data, sorted_features[:n]


def reduce_discretize_dummify_df(dataframe: pd.DataFrame, wanted_columns: list, unwanted_cols_to_disc: list,
                                 discretize_pd_function, bins: int) -> pd.DataFrame:
    dummy_dfs = []
    wanted_col_names = []

    for wanted_col in wanted_columns:
        col_serie = dataframe[wanted_col]  # column array

        # discretize column
        discritized_col = discretize_pd_function(col_serie, bins) \
            if wanted_col not in unwanted_cols_to_disc else col_serie.astype('category')

        # dummify
        dummy_cols = pd.get_dummies(discritized_col)
        dummy_dfs.append(dummy_cols)

        # save wanted col name for each dumified column
        dummy_col_names = dummy_cols.columns
        col_names_to_insert = [wanted_col + str(dummy_col_name) for dummy_col_name in dummy_col_names]
        wanted_col_names += col_names_to_insert

    dummified_df = pd.concat(dummy_dfs, axis=1, ignore_index=True)  # concatenate all dumified columns
    dummified_df.columns = wanted_col_names  # more interpretable column names
    return dummified_df


######
# Frequent patterns / Association rules mining functions
def fp_decreasing_supp(dummified_df: pd.DataFrame, min_frequent_patterns: int, supp_decreasing_rate: float,
                       min_possible_supp: float) -> tuple:
    min_sup = 1.0

    while min_sup > min_possible_supp:
        frequent_itemsets = fp_min_supp(dummified_df, min_sup)[0]

        # enough frequent patterns found
        if len(frequent_itemsets.index) > min_frequent_patterns:
            return frequent_itemsets, min_sup

        # if not decrease min support
        min_sup -= supp_decreasing_rate


def fp_min_supp(dummified_df: pd.DataFrame, min_supp: float) -> tuple:
    return fpgrowth(dummified_df, min_supp, use_colnames=True), min_supp


def association_rules_mining(fp_mining_args: list, min_confidence: float, min_ant_items: int) -> tuple:
    # frequent pattern mining
    frequent_patterns, min_supp = fp_min_supp(*fp_mining_args) \
        if len(fp_mining_args) == 2 else fp_decreasing_supp(*fp_mining_args)

    # association rules mining
    rules = association_rules(frequent_patterns, min_threshold=min_confidence)
    rules = rules[rules['antecedents'].map(len) >= min_ant_items]

    return rules, frequent_patterns, min_supp


def rules_per_target(dummy_df: pd.DataFrame, target: np.ndarray, min_supp: float, min_conf: float,
                     min_ant_items: int) -> list:
    ars_per_target = []
    unique_target = np.unique(target)

    for t in unique_target:
        samples_rows = np.where(target == t)[0]
        target_samples = dummy_df.iloc[samples_rows]
        rules, _, _ = association_rules_mining([target_samples, min_supp], min_conf, min_ant_items)
        ars_per_target.append(rules)

    return ars_per_target


#########
# Aux functions for results
def pretty_print_fsets(freqsets_df: pd.DataFrame, order: bool, n: int) -> None:
    ordered_fsets = freqsets_df.sort_values(['support'], ascending=order).values

    for i in range(n):
        current_fset = ordered_fsets[i]
        supp = current_fset[0]
        itemset = current_fset[1]
        print("Frequent itemset " + str(i) + ": " + str(supp) + " " + str(itemset))


def pretty_print_rules(rules: pd.DataFrame, order_cols: list, order: bool, n: int) -> None:
    ordered_rules = rules.sort_values(order_cols, ascending=order).values

    for i in range(n):
        current_rule = ordered_rules[i]

        # some rule metrics
        supp = str(current_rule[4])
        conf = str(current_rule[5])
        leverage = str(current_rule[7])
        conviction = str(current_rule[8])

        # head and body of the rule
        head = current_rule[0]
        body = current_rule[1]

        print("rule " + str(i) + ": " + str(head) + " -> " + str(body))
        print("[supp=" + supp + ", conf=" + conf + ", leverage=" + leverage + ", conviction=" + conviction + "]\n")


def fsets_per_supp(dummy_df: pd.DataFrame, supports: list) -> None:
    nr_fsets_per_supp = [len(fpgrowth(dummy_df, min_support=supp, use_colnames=True).index) for supp in supports]

    # plot it
    plt.figure()
    plt.title("Frequent itemsets found per min supp")
    plt.xlabel("support")
    plt.ylabel("#frequent itemsets")
    plt.plot(supports, nr_fsets_per_supp, "-o")
    plt.show()


#######
# Pattern mining system function
def pm_system(data: pd.DataFrame, targets: list, k_features: int, selection_measure, discretize_function, bins: int,
              disc_needless_cols, fp_args: list, min_conf: float, min_ant_items: int) -> tuple:
    # feature selection
    _, selected_features = feature_selection_dc(data.values, targets, data.columns, selection_measure, k_features)
    selected_features = [selected_feature[0] for selected_feature in selected_features]

    # transactional encoding
    dummified_df = reduce_discretize_dummify_df(data, selected_features, disc_needless_cols, discretize_function, bins)

    # association rules mining
    rules, frequent_patterns, min_sup = association_rules_mining([dummified_df] + fp_args, min_conf, min_ant_items)

    return selected_features, dummified_df, frequent_patterns, min_sup, rules
