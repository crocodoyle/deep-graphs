import networkx as nx

import csv, os, pickle
import pandas as pd

import numpy as np

import sklearn
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

root_dir = 'E:/brains/graphs/'
targets_file_1 = 'Templeton114.csv'
targets_file_2 = 'Templeton255.csv'


graphtypes = ['AAL', 'desikan', 'CPAC200']


df1 = pd.read_csv(root_dir + targets_file_1)
df2 = pd.read_csv(root_dir + targets_file_2)

n_subjects = 114
n_outputs = 11

ursis = []
graph_features = []

x = np.zeros((n_subjects, 114*114//2 - 114//2), dtype='float32')
y = np.zeros((n_subjects, n_outputs), dtype='float32')


# label_col = df1.loc['URSI']
# print('label col:')
# print(label_col)

r = {}
for graphtype in graphtypes:
    r[graphtype] = []

    y = df1.iloc[1:11]

    for i, graph_filename in enumerate(os.listdir(root_dir + graphtype)):
        ursis.append(graph_filename[4:12])

        graph_data = pickle.load(open(root_dir + graphtype + '/' + graph_filename, 'rb'), encoding='bytes')
        graph_repickled = pickle.dumps(graph_data)

        g = nx.read_gpickle(graph_repickled)
        l = nx.laplacian_matrix(g).to_dense()

        x[i, :] = l[np.triu_indices(114)]

        pls = PLSRegression()


    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]

        pls.fit(x_train, y_train)

        r[graphtype].append(pls.score(x_test, y_test))


print(r)
for graphtype in graphtypes:
    print(graphtype, np.mean(r[graphtype]), np.std(r[graphtype]))





