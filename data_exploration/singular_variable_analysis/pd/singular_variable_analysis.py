from numpy.ma import arange

from data_exploration.singular_variable_analysis.singular_variable_analysis_functions import *
from utils import load_pd
from vis_functions import variables_boxplot

data: pd.DataFrame = load_pd("../../../datasets/pd_speech_features.csv", pop_class=False)

print_shape(data)
print_variable_types(data)
print_missing_values(data)

class_values = [0, 1]

class_balance(data, 'class', class_values)

data, X, y = remove_corr_and_select_k_best(data)

variables_boxplot(data)
plt.show()

data['class'] = y
data_0 = data.loc[data['class'] == 0]
data_1 = data.loc[data['class'] == 1]
data_0.pop('class')
data_1.pop('class')

columns = data.columns
datas = [data_0, data_1]

compare_distributions(datas, 0, range(-6, 7, 2), arange(.0, .35, .05), class_values)
compare_distributions(datas, 1, range(-325000, 0, 75000), arange(.0, .000019, .000002), class_values)
compare_distributions(datas, 2, arange(-.5, .2, .1), range(0, 11, 2), class_values)
compare_distributions(datas, 3, arange(-.8, .3, .2), range(0, 9, 1), class_values)
