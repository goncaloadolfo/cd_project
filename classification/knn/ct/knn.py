from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from utils import *
from vis_functions import multiple_line_chart

DATA_TREATMENT = ['BALANCED', 'NORMALIZE_VARIABLES']


def pre_process(data, typesProcessing):
    for typeProcessing in typesProcessing:
        if typeProcessing == DATA_TREATMENT[1]:
            data = impute_missing_values_second_ds(data)
    return data


def compare_knn(trnX, trnY, tstX, tstY, label):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
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
            recall.append(metrics.recall_score(tstY, prdY, average='micro'))
            accuracy.append(metrics.accuracy_score(tstY, prdY))
        typeDistance.append(d)
        accuracies[d] = accuracy
        recalls[d] = recall

    return typeDistance, accuracies, recalls, nvalues


originalData = pd.read_csv('../../../datasets/secondDataSet.csv', parse_dates=True, infer_datetime_format=True, sep=',')

originalData = get_class_balance_second_ds(originalData)

accuracies = []
recalls = []
titles = []
types = []
testResults = []

for s in DATA_TREATMENT:
    titles.append(s)
    s = [s]
    data = copy(originalData)
    data = pre_process(data, s)
    y: np.ndarray = data.pop('Cover_Type').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    testResults = compare_knn(trnX, trnY, tstX, tstY, labels)
    types.append(testResults[0])
    accuracies.append(testResults[1])
    recalls.append(testResults[2])

nvalues = testResults[3]

for i in range(len(accuracies)):
    print('Data Measures for - ', titles[i])
    print('Accuracy - ', accuracies[i])
    print('Recall - ', recalls[i])

_, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
for i in range(len(accuracies)):
    multiple_line_chart(axs[0, i % 2], nvalues, accuracies[i], str(titles[i]), 'n', 'accuracy', percentage=True)
plt.show()

_, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
for i in range(len(accuracies)):
    multiple_line_chart(axs[0, i % 2], nvalues, recalls[i], str(titles[i]), 'n', 'recall', percentage=True)
plt.show()
