import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neighbors as learn 
import sys
import statistics as statistics

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

DATA_TREATMENT = ['BALANCED', 'NORMALIZE_VARIABLES']

def preProcess(data, typesProcessing):
    for typeProcessing in typesProcessing:
        if(typeProcessing == DATA_TREATMENT[1]):
            data = imputeMissingValuesSecondDT(data)
    return data
            
def compareKNN(trnX, trnY, tstX, tstY, label):
    nvalues = [1, 3, 5,	7,	9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    recalls = {}
    accuracies = {}
    typeDistance = []
    for d in dist:
        recall = []
        accuracy = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            recall.append(metrics.recall_score(tstY, prdY,average='micro'))
            accuracy.append(metrics.accuracy_score(tstY, prdY))
        typeDistance.append(d)
        accuracies[d] = accuracy
        recalls[d] = recall

    return typeDistance, accuracies, recalls, nvalues

def compareNaiveBayes(trnX, trnY, tstX, tstY, labels):
    estimators = {'GaussianNB': GaussianNB(), 
                'BernoulyNB': BernoulliNB()}

    xvalues = []
    yvalues = []
    recalls = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))
        recalls.append(metrics.recall_score(tstY, prdY,average='micro'))

    return xvalues, yvalues, recalls

def mainKNN():
    register_matplotlib_converters()
    originalData = pd.read_csv('datasets/secondDataSet.csv', parse_dates=True, infer_datetime_format=True, sep=',')

    originalData = getClassBalanceSecondDS(originalData)

    accuracies = []
    recalls = []
    titles = []
    types = []
    nvalues = []

    for s in DATA_TREATMENT:
        titles.append(s)
        s = [s]
        data = copy(originalData)
        data = preProcess(data, s)
        y: np.ndarray = data.pop('Cover_Type').values
        X: np.ndarray = data.values
        labels: np.ndarray = pd.unique(y)
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

        testResults = compareKNN(trnX, trnY, tstX, tstY, labels)
        types.append(testResults[0])
        accuracies.append(testResults[1])
        recalls.append(testResults[2])

    nvalues = testResults[3]

    for i in range(len(accuracies)):
        print('Data Measures for - ', titles[i])
        print('Accuracy - ', accuracies[i]) 
        print('Recall - ', recalls[i])

    _, axs = plt.subplots(1,2, figsize=(16, 4), squeeze=False)
    for i in range(len(accuracies)):
        multiple_line_chart(axs[0,i%2], nvalues, accuracies[i], str(titles[i]), 'n', 'accuracy', percentage=True)
    plt.show()

    _, axs = plt.subplots(1,2, figsize=(16, 4), squeeze=False)
    for i in range(len(accuracies)):
        multiple_line_chart(axs[0,i%2], nvalues, recalls[i], str(titles[i]), 'n', 'recall',percentage=True)
    plt.show()

def mainBayes():
    register_matplotlib_converters()
    originalData = pd.read_csv('datasets/secondDataSet.csv', sep=',')

    testResults = []
    titles = []

    originalData = getClassBalanceSecondDS(originalData)

    for s in DATA_TREATMENT:
        titles.append(s)
        s = [s]
        data = copy(originalData)
        data = preProcess(data, s)
        y: np.ndarray = data.pop('Cover_Type').values
        X: np.ndarray = data.values
        labels: np.ndarray = pd.unique(y)
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
        testResults.append(compareNaiveBayes(trnX, trnY, tstX, tstY, labels))

    for i in range(len(testResults)):
        print('Data Measures for - ', titles[i])
        print('Tipo de naive Bayes - ', testResults[i][0])
        print('Accuracy - ', testResults[i][1]) 
        print('Recall - ', testResults[i][2])

    _, axs = plt.subplots(1,2, figsize=(16, 4), squeeze=False)
    j = 0
    for i in range (len(testResults)):
        bar_chart(axs[0,i%2], testResults[i][0], testResults[i][1], str(titles[i]), '','accuracy',percentage=True)
    plt.show()

    _, axs = plt.subplots(1,2, figsize=(16, 4), squeeze=False)
    j = 0
    for i in range (len(testResults)):
        bar_chart(axs[0,i%2], testResults[i][0], testResults[i][2],str(titles[i]), '','recall',percentage=True)
    plt.show()

if __name__ == '__main__':
    mainKNN()
    mainBayes()
