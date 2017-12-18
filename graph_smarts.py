import networkx as nx
from networkx.algorithms.approximation.ramsey import ramsey_R2
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.richclub import rich_club_coefficient

import csv, os, pickle
import pandas as pd

import numpy as np

import sklearn
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression, CCA, PLSCanonical
from sklearn.linear_model import LinearRegression, Lasso, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import normalize, MinMaxScaler

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from deep_stuff import deep_mlp, lr_scheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


root_dir = 'E:/brains/graphs/'
targets_file_1 = 'Templeton114.csv'
targets_file_2 = 'Templeton255.csv'


graphtypes = ['AAL', 'desikan', 'CPAC200', 'HarvardOxford', 'JHU']
rois = [116, 70, 200, 111, 48]


df1 = pd.read_csv(root_dir + targets_file_1, index_col=0)
df2 = pd.read_csv(root_dir + targets_file_2)

n_subjects = 114
n_outputs = 8

ursis = []
graph_features = []

# label_col = df1.loc['URSI']
# print('label col:')
# print(label_col)

ursi_ids = df1.iloc[:, 0]


linreg = LinearRegression(normalize=True)
lasso = Lasso(fit_intercept=True, normalize=True)
ransac = RANSACRegressor()

pls = PLSRegression()
cca = CCA()
pls_ca = PLSCanonical()

rf = RandomForestRegressor(n_estimators=50, n_jobs=4)
gp = GaussianProcessRegressor()
ir = IsotonicRegression()

svr_lin = SVR(kernel='linear')
svr_rbf = SVR()

classifiers = [linreg, lasso, pls, svr_lin, svr_rbf, rf, gp]
classifier_names = ['LR', 'Lasso', 'PLS', 'SVR (lin)', 'SVR (rbf)', 'RF', 'GP']

r = {}
mse = {}
for graphtype in graphtypes:
    r[graphtype] = {}
    mse[graphtype] = {}
    for name in classifier_names:
        r[graphtype][name] = []
        mse[graphtype][name] = []

    mse[graphtype]['deep'] = []
    r[graphtype]['deep'] = []

train_losses = []
val_losses = []

train_percent_error = []
val_percent_error = []

for graphtype, n_roi in zip(graphtypes, rois):
    print('Running analysis for: ', graphtype)

    # y = df1.iloc[:, 1:11].as_matrix()

    # x = np.zeros((n_subjects, (n_roi * n_roi // 2) + (n_roi // 2)), dtype='float32')

    n_features = n_roi + (n_roi // 10) + 1
    x = np.zeros((n_subjects, n_features), dtype='float32')
    y = np.zeros((n_subjects, n_outputs), dtype='float32')
    
    for i, graph_filename in enumerate(os.listdir(root_dir + graphtype)):

        ursi = graph_filename[4:13]
        ursis.append(ursi)

        try:
            targets = np.hstack((df1.loc[ursi, 'CCI'], df1.loc[ursi, :].iloc[4:].as_matrix()))
            y[i, :] = targets

            graph_data = np.load(root_dir + graphtype + '/' + graph_filename)
            # print('graph shape:', graph_data.shape)

            g = nx.Graph(graph_data)

            # feature extraction
            rich_coeff_at_degree = rich_club_coefficient(g, normalized=False)
            rich_keys = list(rich_coeff_at_degree.keys())
            rich_vals = list(rich_coeff_at_degree.values())

            rich_hist, bin_edges = np.histogram(rich_vals, n_roi // 10)
            # print(rich_hist, bin_edges)

            # ramsay = ramsey_R2(g)
            assortivity = degree_assortativity_coefficient(g)

            features = np.sum(graph_data, axis=0) # weighted degree sequence

            # x[i, :] = graph_data[np.triu_indices(n_roi)]
            x[i, :] = np.hstack((features, rich_hist, assortivity))

            # print('nans:', np.sum(np.isnan(graph_data)))
            # print(np.max(x[i, :]), np.min(x[i, :]), np.mean(x[i, :]))

        except KeyError as e:
            print(e)
            i -= 1

    # print(i, 'subjects')

    x_mms, y_mms = MinMaxScaler(), MinMaxScaler()
    x = x_mms.fit_transform(x)
    y = y_mms.fit_transform(y)

    # x = normalize(x, axis=0)

    kf = KFold(n_splits=10)

    # y = y[:, 0]

    print('x:', x.shape)
    print('y:', y.shape)

    for k, (train_index, test_index) in enumerate(kf.split(range(x.shape[0]))):
        print('FOLD:', k+1)
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]

        # print('nans:', np.sum(np.isnan(x_train)))
        # print('infs:', np.sum(np.isinf(x_train)))
        #
        # print('nans:', np.sum(np.isnan(y_train)))
        # print('infs:', np.sum(np.isinf(y_train)))

        model = deep_mlp(n_features)

        model_checkpoint = ModelCheckpoint(root_dir + 'best_model.hdf5', monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=False, mode='min')
        lr_sched = lr_scheduler(model)

        adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
        model.compile('adam', 'mse', metrics=['mean_absolute_percentage_error', 'mean_squared_error'])
        hist = model.fit(x_train, y_train, epochs=100, validation_split=0.1, callbacks=[model_checkpoint, lr_sched])

        train_losses.append(hist.history['loss'])
        val_losses.append(hist.history['val_loss'])
        model.load_weights(root_dir + 'best_model.hdf5')

        predictions = model.predict(x_test)
        actual = y_test

        r2 = r2_score(actual, predictions)
        error = mean_squared_error(actual, predictions)

        r[graphtype]['deep'].append(r2)
        mse[graphtype]['deep'].append(error)

        print('deep', r2)
        for classifier, name in zip(classifiers, classifier_names):

            if isinstance(classifier, type(SVR())):
                classifier.fit(x_train, y_train[:, 0])

                r2 = classifier.score(x_test, y_test[:, 0])
                predictions = classifier.predict(x_test)
                error = mean_squared_error(y_test[:, 0], predictions)

            else:
                classifier.fit(x_train, y_train)

                r2 = classifier.score(x_test, y_test)
                predictions = classifier.predict(x_test)
                error = mean_squared_error(y_test, predictions)

            print(name, r2, error)
            r[graphtype][name].append(r2)
            mse[graphtype][name].append(error)


for graphtype in graphtypes:
    print('Parcelation:', graphtype)
    for name in classifier_names + ['deep']:
        print(name, np.mean(r[graphtype][name]), np.std(r[graphtype][name]))
        print(name, np.mean(mse[graphtype][name]), np.std(mse[graphtype][name]))


# R-SQUARED BOXPLOT
for graphtype in graphtypes:
    scores = []
    score_labels = []

    for name in classifier_names + ['deep']:
        scores.append(r[graphtype][name])
        score_labels.append(name)

    plt.close()
    plt.figure(figsize=(24, 9))
    bplot = plt.boxplot(scores, patch_artist=True, zorder=3)

    plt.xticks(np.arange(1, len(scores)+1), score_labels, rotation=0, horizontalalignment='center', fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=20)
    plt.grid(zorder=0)
    plt.xlim(0, len(scores) + 1)
    plt.ylim(0, 1)

    colors = ['lightcoral', 'pink', 'fuchsia', 'red', 'darkred', 'firebrick', 'm', 'darkblue']

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Classifier', fontsize=24)
    plt.ylabel('$r^2$', fontsize=24)
    plt.tight_layout()

    results_dir = root_dir + '/results/'
    plt.savefig(results_dir + graphtype + '_r2_boxplot.png')

plt.close()

# MEAN SQUARED ERROR BOXPLOT
for graphtype in graphtypes:
    scores = []
    score_labels = []

    for name in classifier_names + ['deep']:
        scores.append(mse[graphtype][name])
        score_labels.append(name)

    plt.close()
    plt.figure(figsize=(24, 9))
    bplot = plt.boxplot(scores, patch_artist=True, zorder=3)

    plt.xticks(np.arange(1, len(scores)+1), score_labels, rotation=0, horizontalalignment='center', fontsize=20)
    # plt.yticks(np.arange(, 1.1, 0.1), fontsize=20)
    plt.grid(zorder=0)
    plt.xlim(0, len(scores) + 1)
    # plt.ylim(0, 1)

    colors = ['lightcoral', 'pink', 'fuchsia', 'red', 'darkred', 'firebrick', 'm', 'darkblue']

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Classifier', fontsize=24)
    plt.ylabel('Mean Squared Error', fontsize=24)
    plt.tight_layout()

    results_dir = root_dir + '/results/'
    plt.savefig(results_dir + graphtype + '_mse_boxplot.png')

plt.close()


plt.figure(figsize=(12, 4))
for train_loss, val_loss in zip(train_losses, val_losses):
    plt.plot(train_loss, color='pink', label='Training')
    plt.plot(val_loss, color='darkred', label='Validation')
    plt.grid(zorder=3)
    plt.legend(shadow=True, fontsize=20)
    plt.tight_layout()
    plt.xlabel('Epoch Number', fontsize=20)
    plt.ylabel('Mean Squared Error (Validation)', fontsize=20)

plt.savefig(results_dir + graphtype + '_loss.png')