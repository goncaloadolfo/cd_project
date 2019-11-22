from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from utils import *
from vis_functions import multiple_line_chart

DATA_TREATMENT = ['NOTHING', 'BALANCE_DATA', 'NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
folds = 10

original_data = load_pd('../../../datasets/pd_speech_features.csv', merge_observations=True, pop_class=False)

testSimilarityResults = []
titles = []


def pre_process(data, types_processing):
    for typeProcessing in types_processing:
        if typeProcessing == DATA_TREATMENT[1]:
            data = get_class_balance(data)
        if typeProcessing == DATA_TREATMENT[2]:
            data = impute_missing_values(data)
        if typeProcessing == DATA_TREATMENT[3]:
            data = erase_correlated_columns(data, 0.6)
    return data


def compare_knn(trnX, trnY, tstX, tstY, label, s):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    recall_values = {}
    for d in dist:
        recalls = []
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            recalls.append(metrics.recall_score(tstY, prdY))
        values[d] = yvalues
        recall_values[d] = recalls

    return nvalues, values, recall_values


for s in DATA_TREATMENT:
    titles.append(s)
    s = [s]
    data = copy(original_data)
    data = pre_process(data, s)
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, y):
        trnX, tstX = X[train_index], X[test_index]
        trnY, tstY = y[train_index], y[test_index]
        testResults = compare_knn(trnX, trnY, tstX, tstY, labels, s)
        testSimilarityResults.append((testResults[0], testResults[1], testResults[2]))

s = ['BALANCE_DATA', 'ERASE_CORRELATED']
titles.append(s)
data = copy(original_data)
data = pre_process(data, s)
y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = y[train_index], y[test_index]
    testResults = compare_knn(trnX, trnY, tstX, tstY, labels, s)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2]))

s = ['BALANCE_DATA', 'NORMALIZE_VARIABLES']
titles.append(s)
data = copy(original_data)
data = pre_process(data, s)
y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = y[train_index], y[test_index]
    testResults = compare_knn(trnX, trnY, tstX, tstY, labels, s)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2]))

s = ['NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
titles.append(s)
data = copy(original_data)
data = pre_process(data, s)
y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = y[train_index], y[test_index]
    testResults = compare_knn(trnX, trnY, tstX, tstY, labels, s)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2]))

s = ['BALANCE_DATA', 'NORMALIZE_VARIABLES', 'ERASE_CORRELATED']
titles.append(s)
data = copy(original_data)
data = pre_process(data, s)
y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)
skf = StratifiedKFold(folds)
for train_index, test_index in skf.split(X, y):
    trnX, tstX = X[train_index], X[test_index]
    trnY, tstY = y[train_index], y[test_index]
    testResults = compare_knn(trnX, trnY, tstX, tstY, labels, s)
    testSimilarityResults.append((testResults[0], testResults[1], testResults[2]))

j = 0
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
    j += 1
    if j == folds:
        recalls = {}
        accuracies = {}
        j = 0

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
_, axs = plt.subplots(3, 3, figsize=(16, 4), squeeze=False)

for i in range(len(testSimilarityResults)):
    if i == 7:
        multiple_line_chart(axs[j, 2], testSimilarityResults[i][0], testSimilarityResults[i][1], titles[titleId],
                            'n', 'accuracy', percentage=True)
    else:
        multiple_line_chart(axs[j, i % 3], testSimilarityResults[i][0], testSimilarityResults[i][1],
                            titles[titleId], 'n', 'accuracy', percentage=True)
    titleId += 1
    if (i + 1) % 3 == 0 and i > 0:
        j += 1
plt.show()

titleId = 0
_, axs = plt.subplots(3, 3, figsize=(16, 4), squeeze=False)
j = 0
for i in range(len(testSimilarityResults)):
    if i == 7:
        multiple_line_chart(axs[j, 2], testSimilarityResults[i][0], testSimilarityResults[i][2], titles[titleId],
                            'n', 'recall', percentage=True)
    else:
        multiple_line_chart(axs[j, i % 3], testSimilarityResults[i][0], testSimilarityResults[i][2],
                            titles[titleId], 'n', 'recall', percentage=True)
    titleId += 1
    if (i + 1) % 3 == 0 and i > 0:
        j += 1
plt.show()
