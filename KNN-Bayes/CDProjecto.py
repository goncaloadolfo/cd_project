import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neighbors as learn 
import numpy as np
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from subprocess import call
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE, RandomOverSampler

from copy import copy
from pandas.plotting import register_matplotlib_converters

from functions import *
from DataTreatment import *

DATA_TREATMENT = ['NOTHING', 'BALANCE_DATA', 'NORMALIZE_VARIABLES', 'ERASE_CORRELATED']

def averageDT(data, repeat):
    i = 0
    j = repeat
    df = pd.DataFrame(columns=data.columns.values)
    while i < data.shape[0]:
        x = data.iloc[i:j]
        mean = x.mean()
        df.loc[int((i)/repeat)] = mean
        j += repeat
        i += repeat
    df.astype({"class": int})
    return df
    
def preProcess(data, typesProcessing):
    for typeProcessing in typesProcessing:
        if(typeProcessing == DATA_TREATMENT[1]):
            data = getClassBalance(data)
        if(typeProcessing == DATA_TREATMENT[2]):
            data = imputeMissingValues(data)
        if(typeProcessing == DATA_TREATMENT[3]):
            data = eraseCorrelatedColumns(data, 0.6)
    return data
            
def compareKNN(trnX, trnY, tstX, tstY, label, s):
    nvalues = [1, 3, 5,	7,	9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    recallValues = {}
    for d in dist:
        recalls = []
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            recalls.append(metrics.recall_score(tstY,prdY))
        values[d] = yvalues
        recallValues[d] = recalls

    return nvalues, values, recallValues

def compareNaiveBayes(trnX, trnY, tstX, tstY, labels):
    estimators = {'GaussianNB': GaussianNB(), 
                'BernoulyNB': BernoulliNB()}

    xvalues = []
    yvalues = []
    recalls = []
    precisions = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        precisions.append(metrics.precision_score (tstY, prdY))
        yvalues.append(metrics.accuracy_score(tstY, prdY))
        recalls.append(metrics.recall_score(tstY,prdY))

    return xvalues, yvalues, recalls, precisions

def mainKNN(folds):
    register_matplotlib_converters()
    originalData = pd.read_csv('datasets/pd_speech_features.csv', parse_dates=True, infer_datetime_format=True, sep=',')


    originalData = averageDT(originalData,3)
    testSimilarityResults = []
    titles = []

    for s in DATA_TREATMENT:
        titles.append(s)
        s = [s]
        data = copy(originalData)
        data = preProcess(data, s)
        y: np.ndarray = data.pop('class').values
        X: np.ndarray = data.values
        labels: np.ndarray = pd.unique(y)
        skf = StratifiedKFold(folds)
        for train_index, test_index in skf.split(X, y):
            trnX, tstX = X[train_index], X[test_index]
            trnY, tstY = y[train_index], y[test_index]
            testResults = compareKNN(trnX, trnY, tstX, tstY, labels,s)
            testSimilarityResults.append((testResults[0],testResults[1],testResults[2]))

    s = ['BALANCE_DATA', 'ERASE_CORRELATED']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        testResults = compareKNN(trnX, trnY, tstX, tstY, labels,s)
        testSimilarityResults.append((testResults[0],testResults[1],testResults[2]))

        

    s = ['BALANCE_DATA', 'NORMALIZE_VARIABLES']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        testResults = compareKNN(trnX, trnY, tstX, tstY, labels,s)
        testSimilarityResults.append((testResults[0],testResults[1],testResults[2]))

    s = ['NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        testResults = compareKNN(trnX, trnY, tstX, tstY, labels,s)
        testSimilarityResults.append((testResults[0],testResults[1],testResults[2]))


    s = ['BALANCE_DATA','NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        testResults = compareKNN(trnX, trnY, tstX, tstY, labels,s)
        testSimilarityResults.append((testResults[0],testResults[1],testResults[2]))

    j=0
    accuraciesChebychev = []
    recallsChebychev = []
    accuraciesManhattan = []
    recallsManhattan = []
    accuraciesEuclidean = []
    recallsEuclidean = []
    new_test = []
    for i in range(len(testSimilarityResults)):
        recallsChebychev.append(testSimilarityResults[i][1]['chebyshev'])
        accuraciesChebychev.append(testSimilarityResults[i][2]['chebyshev'])
        recallsManhattan.append(testSimilarityResults[i][1]['manhattan'])
        accuraciesManhattan.append(testSimilarityResults[i][2]['manhattan'])
        recallsEuclidean.append(testSimilarityResults[i][1]['euclidean'])
        accuraciesEuclidean.append(testSimilarityResults[i][2]['euclidean'])
        j+=1
        if(j == folds):
            recalls = {}
            accuracies = {}
            j=0

            recallsChebychev = np.array(recallsChebychev)
            accuraciesChebychev = np.array(accuraciesChebychev)
            recallsManhattan = np.array(recallsManhattan)
            accuraciesManhattan = np.array(accuraciesManhattan)
            recallsEuclidean = np.array(recallsEuclidean)
            accuraciesEuclidean = np.array(accuraciesEuclidean)

            recalls['chebychev'] = np.mean(recallsChebychev, axis=0)
            accuracies['chebychev'] = np.mean(accuraciesChebychev, axis=0)
            recalls['manhattan'] = np.mean(recallsManhattan, axis=0)
            accuracies['manhattan'] = np.mean(accuraciesManhattan, axis=0)
            recalls['euclidean'] = np.mean(recallsEuclidean, axis=0)
            accuracies['euclidean'] = np.mean(accuraciesEuclidean, axis=0)

            accuraciesChebychev = []
            recallsChebychev = []
            accuraciesManhattan = []
            recallsManhattan = []
            accuraciesEuclidean = []
            recallsEuclidean = []
            
            new_test.append((testSimilarityResults[0][0], accuracies, recalls))

    j = 0

    testSimilarityResults = new_test
    titleId = 0
    _, axs = plt.subplots(3,3, figsize=(16, 4), squeeze=False)

    for i in range(len(testSimilarityResults)):
        if(i == 7):
            multiple_line_chart(axs[j,2],testSimilarityResults[i][0], testSimilarityResults[i][1], titles[titleId], 'n', 'accuracy', percentage=True)
        else :
            multiple_line_chart(axs[j,i%3],testSimilarityResults[i][0], testSimilarityResults[i][1], titles[titleId], 'n', 'accuracy', percentage=True)
        titleId += 1
        if((i+1)%3==0 and i>0):
            j+=1
    plt.show()

    titleId = 0
    _, axs = plt.subplots(3,3, figsize=(16, 4), squeeze=False)
    j = 0
    for i in range(len(testSimilarityResults)):
        if(i == 7):
            multiple_line_chart(axs[j,2],testSimilarityResults[i][0], testSimilarityResults[i][2], titles[titleId], 'n', 'recall', percentage=True)
        else :
            multiple_line_chart(axs[j,i%3],testSimilarityResults[i][0], testSimilarityResults[i][2], titles[titleId], 'n', 'recall', percentage=True)
        titleId += 1
        if((i+1)%3==0 and i>0):
            j+=1
    plt.show()

def mainBayes(folds):
    register_matplotlib_converters()
    originalData = pd.read_csv('datasets/pd_speech_features.csv', parse_dates=True, infer_datetime_format=True, sep=',')

    originalData = averageDT(originalData,3)
    titles = []
    testSimilarityResults = []

    for s in DATA_TREATMENT:
        titles.append(s)
        s = [s]
        data = copy(originalData)
        data = preProcess(data, s)
        Y: np.ndarray = data.pop('class').values
        X: np.ndarray = data.values
        labels: np.ndarray = pd.unique(Y)
        skf = StratifiedKFold(folds)
        for train_index, test_index in skf.split(X, Y):
            trnX, tstX = X[train_index], X[test_index]
            trnY, tstY = Y[train_index], Y[test_index]
            testResults = compareNaiveBayes(trnX, trnY, tstX, tstY, labels)
            testSimilarityResults.append((testResults[0], testResults[1], testResults[2],testResults[3]))

    s = ['BALANCE_DATA', 'ERASE_CORRELATED']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    Y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(Y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, Y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = Y[train_index], Y[test_index]
        testResults = compareNaiveBayes(trnX, trnY, tstX, tstY, labels)
        testSimilarityResults.append((testResults[0], testResults[1], testResults[2],testResults[3]))
        

    s = ['BALANCE_DATA', 'NORMALIZE_VARIABLES']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    Y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(Y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, Y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = Y[train_index], Y[test_index]
        testResults = compareNaiveBayes(trnX, trnY, tstX, tstY, labels)
        testSimilarityResults.append((testResults[0], testResults[1], testResults[2],testResults[3]))

    s = ['NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    Y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(Y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, Y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = Y[train_index], Y[test_index]
        testResults = compareNaiveBayes(trnX, trnY, tstX, tstY, labels)
        testSimilarityResults.append((testResults[0], testResults[1], testResults[2],testResults[3]))


    s = ['BALANCE_DATA','NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
    titles.append(s)
    data = copy(originalData)
    data = preProcess(data, s)
    Y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(Y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, Y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = Y[train_index], Y[test_index]
        testResults = compareNaiveBayes(trnX, trnY, tstX, tstY, labels)
        testSimilarityResults.append((testResults[0], testResults[1], testResults[2],testResults[3]))

    testSimilarityResults = np.array(testSimilarityResults)

    j = 0
    titulo = 0
    meanTestResults = []
    gaussianAccuracy = 0
    bernoullyAccuracy = 0 
    gaussianRecall = 0
    bernoulyRecall = 0
    gaussianPrecision = 0
    bernoullyPrecision = 0
    for i in range(len(testSimilarityResults)):

        gaussianAccuracy += float(testSimilarityResults[i][1][0])
        bernoullyAccuracy += float(testSimilarityResults[i][1][1])
        gaussianRecall += float(testSimilarityResults[i][2][0])
        bernoulyRecall += float(testSimilarityResults[i][2][1])
        gaussianPrecision += float(testSimilarityResults[i][3][0])
        bernoullyPrecision += float(testSimilarityResults[i][3][1])

        j += 1
        if(j == folds):
            x = []
            x.append(testSimilarityResults[0][0])
            x.append([gaussianAccuracy/folds,bernoullyAccuracy/folds])
            x.append([gaussianRecall/folds,bernoulyRecall/folds])
            x.append([gaussianPrecision/folds, bernoullyPrecision/folds])        
            gaussianAccuracy = 0
            bernoullyAccuracy = 0
            gaussianRecall = 0
            bernoulyRecall = 0
            gaussianPrecision = 0
            bernoullyPrecision = 0
  
            meanTestResults.append(x)
            j= 0
            titulo+=1

    testSimilarityResults = meanTestResults

    _, axs = plt.subplots(3,3, figsize=(16, 4), squeeze=False)
    titleId = 0
    for i in range(len(testSimilarityResults)):
        if(i == 7):
            bar_chart(axs[j, 2] ,testSimilarityResults[i][0], testSimilarityResults[i][1], titles[titleId], '', 'accuracy', percentage=True)
        else :
            bar_chart(axs[j, i%3], testSimilarityResults[i][0], testSimilarityResults[i][1], titles[titleId], '', 'accuracy', percentage=True)
        titleId += 1
        if((i+1)%3==0 and i>0):
            j+=1
    plt.show()

    _, axs = plt.subplots(3,3, figsize=(16, 4), squeeze=False)

    j = 0
    titleId = 0
    for i in range(len(testSimilarityResults)):
        if(i == 7):
            bar_chart(axs[j, 2] ,testSimilarityResults[i][0], testSimilarityResults[i][2], str(titles[titleId]), '', 'recall', percentage=True)
        else :
            bar_chart(axs[j, i%3], testSimilarityResults[i][0], testSimilarityResults[i][2], str(titles[titleId]), '', 'recall', percentage=True)
        titleId += 1
        if((i+1)%3==0 and i>0):
            j+=1
    plt.show()

    _, axs = plt.subplots(3,3, figsize=(16, 4), squeeze=False)

    j = 0
    titleId = 0
    for i in range(len(testSimilarityResults)):
        if(i == 7):
            bar_chart(axs[j, 2] ,testSimilarityResults[i][0], testSimilarityResults[i][3], str(titles[titleId]), '', 'precision', percentage=True)
        else :
            bar_chart(axs[j, i%3], testSimilarityResults[i][0], testSimilarityResults[i][3], str(titles[titleId]), '', 'precision', percentage=True)
        titleId += 1
        if((i+1)%3==0 and i>0):
            j+=1
    plt.show()

if __name__ == '__main__':
    mainKNN(10)
    mainBayes(10)
