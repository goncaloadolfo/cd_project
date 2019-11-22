from numpy.ma import arange

from data_exploration.singular_variable_analysis.singular_variable_analysis_functions import *
from utils import load_pd
from vis_functions import variables_boxplot

data: pd.DataFrame = pd.read_csv("../../../datasets/secondDataSet.csv")

print_shape(data)
print_variable_types(data)
print_missing_values(data)

class_values = [1, 2, 3, 4, 5, 6, 7]

class_balance(data, 'Cover_Type', class_values)

data, X, y = remove_corr_and_select_k_best(data, class_name='Cover_Type')

variables_boxplot(data)
plt.show()

data['class'] = y
data_1 = data.loc[data['class'] == 1]
data_2 = data.loc[data['class'] == 2]
data_3 = data.loc[data['class'] == 3]
data_4 = data.loc[data['class'] == 4]
data_5 = data.loc[data['class'] == 5]
data_6 = data.loc[data['class'] == 6]
data_7 = data.loc[data['class'] == 7]
data_1.pop('class')
data_2.pop('class')
data_3.pop('class')
data_4.pop('class')
data_5.pop('class')
data_6.pop('class')
data_7.pop('class')

columns = data.columns
datas = [data_1, data_2, data_3, data_4, data_5, data_6, data_7]

compare_distributions(datas, 0, range(1750, 4250, 750), arange(.0, .007, .001), class_values)
compare_binary_distributions(datas, 1, range(0, 300100, 50000), class_values)
compare_binary_distributions(datas, 2, range(0, 300100, 50000), class_values)
compare_binary_distributions(datas, 3, range(0, 300100, 50000), class_values)
