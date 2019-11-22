from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from utils import *
from vis_functions import bar_chart

DATA_TREATMENT = ['BALANCED', 'NORMALIZE_VARIABLES']


def pre_process(data, typesProcessing):
    for typeProcessing in typesProcessing:
        if typeProcessing == DATA_TREATMENT[1]:
            data = impute_missing_values_second_ds(data)
    return data


def compare_naive_bayes(trnX, trnY, tstX, tstY, labels):
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
        recalls.append(metrics.recall_score(tstY, prdY, average='micro'))

    return xvalues, yvalues, recalls


originalData = pd.read_csv('../../../datasets/secondDataSet.csv', sep=',')

testResults = []
titles = []

originalData = get_class_balance_second_ds(originalData)

for s in DATA_TREATMENT:
    titles.append(s)
    s = [s]
    data = copy(originalData)
    data = pre_process(data, s)
    y: np.ndarray = data.pop('Cover_Type').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    testResults.append(compare_naive_bayes(trnX, trnY, tstX, tstY, labels))

for i in range(len(testResults)):
    print('Data Measures for - ', titles[i])
    print('Tipo de naive Bayes - ', testResults[i][0])
    print('Accuracy - ', testResults[i][1])
    print('Recall - ', testResults[i][2])

_, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
j = 0
for i in range(len(testResults)):
    bar_chart(axs[0, i % 2], testResults[i][0], testResults[i][1], str(titles[i]), '', 'accuracy', percentage=True)
plt.show()

_, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
j = 0
for i in range(len(testResults)):
    bar_chart(axs[0, i % 2], testResults[i][0], testResults[i][2], str(titles[i]), '', 'recall', percentage=True)
plt.show()
