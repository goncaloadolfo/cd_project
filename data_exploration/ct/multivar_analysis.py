"""
Multivariate analysis for Cover type Data Set.
"""

# libs
import numpy as np
import matplotlib.pyplot as plt

# own libs
from data_exploration.multi_analysis_functions import draw_heatmap, correlation_histogram
from utils import undersampling_ct
from vis_functions import bar_chart, scatter_plot


#####
# under sampling cover type data set
ct_data = undersampling_ct("../../datasets/secondDataSet.csv")
cover_types = np.unique(ct_data['Cover_Type'])

#####
# show data balance after under sampling
ct_per_area = []
area_col_names = ['Wilderness_Area0', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3']
area_names = ['Rawah', 'Neota', 'Comanche Peak', 'Cache la Poudre']

for area_i in range(4):
    df_area = ct_data[ct_data[area_col_names[area_i]] == 1]
    area_target = df_area['Cover_Type'].values
    target_counting = [np.sum(area_target == target) for target in cover_types]
    ct_per_area.append(target_counting)

fig, axes = plt.subplots(2, 2)
xvalues_charts = cover_types.astype(np.str)
bar_chart(axes[0, 0], xvalues_charts, ct_per_area[0], area_names[0], "cover type", "#samples")
bar_chart(axes[0, 1], xvalues_charts, ct_per_area[1], area_names[1], "cover type", "#samples")
bar_chart(axes[1, 0], xvalues_charts, ct_per_area[2], area_names[2], "cover type", "#samples")
bar_chart(axes[1, 1], xvalues_charts, ct_per_area[3], area_names[3], "cover type", "#samples")

#####
# correlation between measure variables
targets = ct_data.pop('Cover_Type')
measures_df = ct_data.iloc[:, :10]

corr_fig, corr_axes = plt.subplots(1, 2)
draw_heatmap(measures_df.corr(), measures_df.columns, measures_df.columns, True, 'Greens',
             "Measures Correlation", "variable", "variable", corr_axes[0])
correlation_histogram("Correlation histogram", measures_df.corr().values, corr_axes[1])

#####
# some scatter plots between measure variables
target_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'black']
focus_var = ct_data.columns[0]
other_vars = ct_data.columns[1: 7]
scatter_fig, scatter_axes = plt.subplots(2, 3)

focus_var_values = ct_data[focus_var]
for other_var_i in range(6):
    other_var = other_vars[other_var_i]
    other_var_values = ct_data[other_var]

    row = other_var_i // 3
    col = other_var_i % 3

    for cover_type_i in range(7):
        curr_ct = cover_types[cover_type_i]
        xvalues = focus_var_values[targets == curr_ct]
        yvalues = other_var_values[targets == curr_ct]
        color = target_colors[cover_type_i]
        scatter_plot(scatter_axes[row, col], xvalues, yvalues, "", focus_var, other_var, curr_ct, color, alpha=0.5)

    scatter_axes[row, col].legend()

plt.tight_layout()
plt.show()
