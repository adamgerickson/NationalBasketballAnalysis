import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.metrics import recall_score
import itertools

import sys

# get all data
df = pd.read_csv('test_strippeddown.csv')

# fill in missing data
df = df.fillna(0)


# remove test data
# test = df[20:30]
# df.drop(df.index[20:30], inplace=True)

# # split test data
# test_labels = test[['allstar']]
# test_names = test[['name']]
# del test['allstar']
# del test['name']


labels = df[['allstar']]
names = df[['name']]
del df['allstar']
del df['name']


# change shape of labels to (n_samples, ).
labels = np.ravel(labels)

classifier = SVC(kernel='rbf',gamma=0.5)
classifier.fit(df, labels)


# # try out model selection
parameters = {} #{'kernel' : ('linear')} # {'criterion' : ('gini', 'entropy'), 'splitter' : ('best', 'random')}
model = GridSearchCV(classifier, parameters, scoring='f1_micro')
model.fit(df, labels)

# print(test_labels)
# predicted = model.predict_proba(test)

# predicted_0_true = 0
# predicted_0_false = 0
# predicted_1_true = 0
# predicted_1_false = 0
# correct = 0
# total = 0

# # threshold probability value for being put in the allstar class (if prob > threshold, assign as allstar)
# threshold = 0.3

# for prediction, real, name in zip(predicted, test_labels['allstar'], test_names['name']):
#     print(name, real, prediction)
#     total += 1

#     prediction = (1 if prediction[1] >= threshold else -1)
#     print(prediction)

#     if prediction == real:
#         correct += 1
#         if prediction == -1:
#             predicted_0_true += 1
#         else:
#             predicted_1_true += 1
#     else:
#         if prediction == -1:
#             predicted_0_false += 1
#         else:
#             predicted_1_false += 1




# print('percent correct: {}'.format(correct / float(total)))
# print('predicted 0: True {} False {}\npredicted 1: True {} False {}'.format(predicted_0_true, predicted_0_false, predicted_1_true, predicted_1_false))

# print(model.best_estimator_)

################################################################################################################################

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from scipy.io import loadmat as load
from numpy import argsort, reshape, transpose, array, zeros,delete
from matplotlib.pyplot import imshow, xlabel, ylabel, title, figure, savefig
from numpy.random import permutation, seed
from pydotplus import graph_from_dot_data
from sklearn.externals.six import StringIO

x = df
c = labels

correct = 0
w = h = 2

k = 10
correct = 0
conf = [[0 for x_ in range(w)] for y_ in range(h)]
seed(3)
x_ = permutation(x[:])
seed(3)
Y = permutation(c[:])

split_size = len(Y) * 1.0 / k
folds = [i for i in range(k) for _ in range(int(split_size))]
num_bigger_sets = len(Y) - len(folds)
if num_bigger_sets != 0:
    first = [i for i in range(k - num_bigger_sets) for _ in range(int(split_size))]
    second = [i + (k - num_bigger_sets) for i in range(num_bigger_sets) for _ in range(int(split_size + 1))] 
    folds = first + second



for j in range(k):
    X_train = [row for row, f in zip(x_,folds) if f != j]
    Y_train = [val for val, f in zip(Y,folds) if f != j]

    X_test = [row for row, f in zip(x_,folds) if f == j]
    Y_test = [val for val, f in zip(Y,folds) if f == j]

    M = SVC(kernel='rbf', gamma=0.5)
    # M = RandomForestClassifier()
    M = M.fit(X_train, Y_train)

    predicted = M.predict(X_test) 
    for i in range(len(predicted)):
        if predicted[i] == Y_test[i]:
            correct += 1
        conf[(0 if Y_test[i] == -1 else 1)][(0 if predicted[i] == -1 else 1)] += 1

print conf
print k, "fold accuracy:", correct * 1.0 / len(c)

col1total = float(conf[0][0] + conf[1][0])
col2total = float(conf[0][1] + conf[1][1])

if col2total > 0:
    print('num correct that were predicted allstars: {} / {} = {}'.format(conf[1][1], col2total, conf[1][1] / col2total))
print(model.best_params_)


num_classes = 2
classes = range(num_classes)
class_names = ["Normal", "All-star"]
title = str(k) + "-fold Confusion Matrix - SVC rbf kernel"

# normalize the conf matrix
if col2total > 0:
    conf_normalized = [[conf[0][0] / col1total, conf[0][1] / col2total], [conf[1][0] / col1total, conf[1][1] / col2total]]
else:
    conf_normalized = conf
thresh = max([max(conf[0]), max(conf[1])]) / 2
imshow(conf_normalized,interpolation='nearest', cmap=plt.cm.Greens)
for i, j in itertools.product(classes, classes):
        plt.text(j, i, conf[i][j],
                 horizontalalignment="center",
                 color="white" if conf[i][j] > thresh else "black")

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title(title)
savefig(title)
plt.show()