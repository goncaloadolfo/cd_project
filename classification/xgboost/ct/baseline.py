"""
Context: Cover Type Decease Data Set
XGBoost results without pre-processing and parameter tuning.
"""

# libs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# own libs
from utils import load_and_undersample_ct
from vis_functions import heatmap


def print_results(prefix: str, acc: float, prec: float, rec: float) -> None:
    print(prefix)
    print("Accuracy={}, Precision={}, Recall={}".format(acc, prec, rec))


# load data
data = load_and_undersample_ct("../../../datasets/secondDataSet.csv")
target = data.pop("Cover_Type")
data = data.values

# train/test split
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, shuffle=True, random_state=40,
                                                    stratify=target)

# predict train set and test set
xgboost = XGBClassifier()
xgboost.fit(x_train, y_train)
y_predicted = xgboost.predict(x_test)
train_predictions = xgboost.predict(x_train)

# train metrics
acc_train = accuracy_score(y_train, train_predictions)
prec_train = precision_score(y_train, train_predictions, average='macro')
rec_train = recall_score(y_train, train_predictions, average='macro')

# test metrics
acc_test = accuracy_score(y_test, y_predicted)
prec_test = precision_score(y_test, y_predicted, average='macro')
rec_test = recall_score(y_test, y_predicted, average='macro')

# print results
print_results("Evaluation on training set", acc_train, prec_train, rec_train)
print_results("Evaluation on test set", acc_test, prec_test, rec_test)

cm = confusion_matrix(y_test, y_predicted)
plt.figure()
heatmap(plt.gca(), cm, "Test Confusion Matrix", "predicted class", "true class")

plt.show()
