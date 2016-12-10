from sklearn.tree import DecisionTreeClassifier, export_graphviz
from scipy.io import loadmat as load
from numpy import argsort, reshape, transpose, array, zeros,delete
from matplotlib.pyplot import imshow, xlabel, ylabel, title, figure, savefig
from numpy.random import permutation, seed
from pydotplus import graph_from_dot_data
from sklearn.externals.six import StringIO 


x = X[:]
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

    M = svm.SVC(gamma = clf.best_params_['gamma'], C = clf.best_params_['C'], kernel = clf.best_params_['kernel'],probability=True)
    M = M.fit(X_train, Y_train)

    predicted = M.predict(X_test) 
    for i in range(len(predicted)):
        if predicted[i] == Y_test[i]:
            correct += 1
        conf[Y_test[i]][predicted[i]] += 1

print conf
print k, "fold accuracy:", correct * 1.0 / len(c)
imshow(conf,interpolation='nearest')
title(str(k) + "-fold Confusion Matrix")
ylabel("Actual Class")
xlabel("Predicted Class")
savefig(str(k) + 'fold_conf.png')