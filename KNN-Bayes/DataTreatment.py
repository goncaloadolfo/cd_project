import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE, RandomOverSampler

from copy import copy

from functions import *

def dummify(df, cols_to_dummify):
    one_hot_encoder = OneHotEncoder(sparse=False)

    for var in cols_to_dummify:
        one_hot_encoder.fit(df[var].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([var])
        transformed_data = one_hot_encoder.transform(df[var].values.reshape(-1, 1))
        df = pd.concat((df, pd.DataFrame(transformed_data, columns=feature_names)), 1)
        df.pop(var)
    return df
  

def imputeMissingValues(original):

    cols_nr = original.select_dtypes(include='number')
    cols_sb = original.select_dtypes(include='category')

    imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
    imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
    if(len(cols_nr.T)>0):
        df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
    if(len(cols_sb.T)>0):
        df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

    if(len(cols_nr.T)>0 and len(cols_sb.T)>0):
        data = df_nr.join(df_sb, how='right')
        data.describe(include='all')
    else:
        data = df_nr
    original = data
    aux = original.pop('class').values
    data = Normalizer().fit_transform(original)
    data = pd.DataFrame(data, columns= original.columns)
    data.insert(data.shape[1], 'class', aux)
    return data

def imputeMissingValuesSecondDT(original):
    cols_nr = original.select_dtypes(include='number')
    cols_sb = original.select_dtypes(include='category')

    imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
    imp_sb = SimpleImputer(strategy='most_frequent', missing_values='', copy=True)
    if(len(cols_nr.T)>0):
        df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)
    if(len(cols_sb.T)>0):
        df_sb = pd.DataFrame(imp_sb.fit_transform(cols_sb), columns=cols_sb.columns)

    if(len(cols_nr.T)>0 and len(cols_sb.T)>0):
        data = df_nr.join(df_sb, how='right')
        data.describe(include='all')
    else:
        data = df_nr
    original = data
    aux = original.pop('Cover_Type').values
    data = Normalizer().fit_transform(original)
    data = pd.DataFrame(data, columns= original.columns)
    data.insert(data.shape[1], 'Cover_Type', aux)
    return data


def getClassBalance(data):
    unbal = copy(data)
    target_count = unbal['class'].value_counts()
    #plt.figure()
    #plt.title('Class balance')
    #plt.bar(target_count.index, target_count.values)
    #plt.show()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class:', target_count[ind_min_class])
    print('Majority class:', target_count[1-ind_min_class])
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

    RANDOM_STATE = 42
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}

    df_class_min = unbal[unbal['class'] == min_class]
    df_class_max = unbal[unbal['class'] != min_class] 

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]

    smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
    y = unbal.pop('class').values
    X = unbal.values
    smote_x, smote_y = smote.fit_sample(X, y)
    newData = []
    for i in range(len(smote_x)):
        newData.append(np.append(smote_x[i],smote_y[i]))
    
    data = pd.DataFrame(data=newData, columns = data[:0].columns)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]

    # plt.figure()
    # multiple_bar_chart(plt.gca(), 
    #                         [target_count.index[ind_min_class], target_count.index[1-ind_min_class]], 
    #                         values, 'Target', 'frequency', 'Class balance')
    # plt.show()
    return data 

def getClassBalanceSecondDS(data):
    unbal = copy(data)
    target_count = unbal['Cover_Type'].value_counts()
    #plt.figure()
    #plt.title('Class balance')
    #plt.bar(target_count.index, target_count.values)
    #plt.show()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    df_class_min = unbal[unbal['Cover_Type'] == min_class]

    aux = df_class_min
    for i in unbal['Cover_Type'].unique():
        if(i!=min_class):
            newList = pd.DataFrame(unbal[unbal['Cover_Type'] == i])
            aux = pd.concat([aux,newList.sample(len(df_class_min))])

    data = pd.DataFrame(data=aux, columns = data[:0].columns)
    return data 


def eraseCorrelatedColumns (data, threshold):
    corr_mtx = data.corr()
    keyId = 0
    valueId = 0 
    keys = corr_mtx.keys()
    correlacionados = []
    for i in corr_mtx.values:
        keyId = 0 
        for j in i:
            if(abs(j)>threshold and keyId != valueId):
                value = keys[valueId], keys [keyId]
                iValue = keys[keyId], keys [valueId]
                if(not (iValue in correlacionados)):
                    correlacionados.append (value)
            keyId += 1
        valueId += 1


    for v in correlacionados:
        if(v[1] in data.keys() and v[0] in data.keys()):
            data = data.drop(columns = [v[1]])
    return data
