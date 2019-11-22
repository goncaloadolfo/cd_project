from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from utils import *
from vis_functions import bar_chart

DATA_TREATMENT = ['NOTHING', 'BALANCE_DATA', 'NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
folds = 10

originalData = load_pd('../../../datasets/pd_speech_features.csv', merge_observations=True, pop_class=False)

titles = []
testSimilarityResults = []


def pre_process(data, types_processing):
    for typeProcessing in types_processing:
        if typeProcessing == DATA_TREATMENT[1]:
            data = get_class_balance(data)
        if typeProcessing == DATA_TREATMENT[2]:
            data = impute_missing_values(data)
        if typeProcessing == DATA_TREATMENT[3]:
            data = erase_correlated_columns(data, 0.6)
    return data


def compare_naive_bayes(trnX, trnY, tstX, tstY, labels):
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
        precisions.append(metrics.precision_score(tstY, prdY))
        yvalues.append(metrics.accuracy_score(tstY, prdY))
        recalls.append(metrics.recall_score(tstY, prdY))

    return xvalues, yvalues, recalls, precisions


for s in DATA_TREATMENT:
    titles.append(s)
    s = [s]
    data = copy(originalData)
    data = pre_process(data, s)
    Y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(Y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, Y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = Y[train_index], Y[test_index]
        testResults = compare_naive_bayes(trnX, trnY, tstX, tstY, labels)
        testSimilarityResults.append((testResults[0], testResults[1], testResults[2], testResults[3]))

s = ['BALANCE_DATA', 'ERASE_CORRELATED']
titles.append(s)
data = copy(originalData)
data = pre_process(data, s)
Y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(Y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, Y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = Y[train_index], Y[test_index]
    testResults = compare_naive_bayes(trnX, trnY, tstX, tstY, labels)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2], testResults[3]))

s = ['BALANCE_DATA', 'NORMALIZE_VARIABLES']
titles.append(s)
data = copy(originalData)
data = pre_process(data, s)
Y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(Y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, Y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = Y[train_index], Y[test_index]
    testResults = compare_naive_bayes(trnX, trnY, tstX, tstY, labels)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2], testResults[3]))

s = ['NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
titles.append(s)
data = copy(originalData)
data = pre_process(data, s)
Y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(Y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, Y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = Y[train_index], Y[test_index]
    testResults = compare_naive_bayes(trnX, trnY, tstX, tstY, labels)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2], testResults[3]))

s = ['BALANCE_DATA', 'NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
titles.append(s)
data = copy(originalData)
data = pre_process(data, s)
Y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(Y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, Y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = Y[train_index], Y[test_index]
    testResults = compare_naive_bayes(trnX, trnY, tstX, tstY, labels)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2], testResults[3]))

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
    if j == folds:
        x = [testSimilarityResults[0][0], [gaussianAccuracy / folds, bernoullyAccuracy / folds],
             [gaussianRecall / folds, bernoulyRecall / folds], [gaussianPrecision / folds, bernoullyPrecision / folds]]
        gaussianAccuracy = 0
        bernoullyAccuracy = 0
        gaussianRecall = 0
        bernoulyRecall = 0
        gaussianPrecision = 0
        bernoullyPrecision = 0

        meanTestResults.append(x)
        j = 0
        titulo += 1

testSimilarityResults = meanTestResults

_, axs = plt.subplots(3, 3, figsize=(16, 4), squeeze=False)
titleId = 0
for i in range(len(testSimilarityResults)):
    if i == 7:
        bar_chart(axs[j, 2], testSimilarityResults[i][0], testSimilarityResults[i][1], titles[titleId], '',
                  'accuracy', percentage=True)
    else:
        bar_chart(axs[j, i % 3], testSimilarityResults[i][0], testSimilarityResults[i][1], titles[titleId], '',
                  'accuracy', percentage=True)
    titleId += 1
    if (i + 1) % 3 == 0 and i > 0:
        j += 1
plt.show()

_, axs = plt.subplots(3, 3, figsize=(16, 4), squeeze=False)

j = 0
titleId = 0
for i in range(len(testSimilarityResults)):
    if i == 7:
        bar_chart(axs[j, 2], testSimilarityResults[i][0], testSimilarityResults[i][2], str(titles[titleId]), '',
                  'recall', percentage=True)
    else:
        bar_chart(axs[j, i % 3], testSimilarityResults[i][0], testSimilarityResults[i][2], str(titles[titleId]), '',
                  'recall', percentage=True)
    titleId += 1
    if (i + 1) % 3 == 0 and i > 0:
        j += 1
plt.show()

_, axs = plt.subplots(3, 3, figsize=(16, 4), squeeze=False)

j = 0
titleId = 0
for i in range(len(testSimilarityResults)):
    if i == 7:
        bar_chart(axs[j, 2], testSimilarityResults[i][0], testSimilarityResults[i][3], str(titles[titleId]), '',
                  'precision', percentage=True)
    else:
        bar_chart(axs[j, i % 3], testSimilarityResults[i][0], testSimilarityResults[i][3], str(titles[titleId]), '',
                  'precision', percentage=True)
    titleId += 1
    if (i + 1) % 3 == 0 and i > 0:
        j += 1
plt.show()
