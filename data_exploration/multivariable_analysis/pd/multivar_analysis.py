"""
Multivariate analysis for PD data set.
"""

import matplotlib.pyplot as plt

from data_exploration.multivariable_analysis.multivar_analysis_functions import multi_analysis
from utils import load_pd

# globals
PARKINSON_ATTR_GROUPS = {
    "Baseline Features": (2, 22),
    "Intensity Parameters": (23, 25),
    "Formant Frequencies": (26, 29),
    "Bandwidth Parameters": (30, 33),
    "Vocal Fold": (34, 55),
    "MFCC": (56, 139),
    "Wavelet Features": (140, 321),
    "TQWT Features": (322, 753)
}

# load parkison data
data, X, y = load_pd("../../datasets/pd_speech_features.csv")

multi_analysis(data, y, PARKINSON_ATTR_GROUPS)

plt.show()
