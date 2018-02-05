import networkx as nx
from networkx.algorithms.approximation.ramsey import ramsey_R2
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.richclub import rich_club_coefficient

import csv, os, pickle
import pandas as pd

import numpy as np
import time

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


start_time = time.time()

root_dir = 'E:/brains/graphs/'
targets_file_1 = 'Templeton114.csv'

targets_file_2 = 'Templeton255.csv'
test_dir = root_dir + 'Templeton255/'

datasets = ['Templeton114', 'Templeton255']
# graphtypes = ['AAL', 'desikan', 'CPAC200', 'HarvardOxford', 'JHU']
# rois = [116, 70, 200, 111, 48]
graphtypes = ['desikan', 'CPAC200', 'HarvardOxford']
rois = [70, 200, 111]

df1 = pd.read_csv(root_dir + targets_file_1, index_col=0)
df2 = pd.read_csv(root_dir + targets_file_2, index_col=0)

n_subjects = 114
n_subjects2 = 255
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

prediction_targets = ['all', 'CCI']

r = {}
mse = {}

for dataset in datasets:
    r[dataset] = {}
    mse[dataset] = {}
    for targets in prediction_targets:
        r[dataset][targets] = {}
        mse[dataset][targets] = {}
        for graphtype in graphtypes:
            r[dataset][targets][graphtype] = {}
            mse[dataset][targets][graphtype] = {}
            for name in classifier_names:
                r[dataset][targets][graphtype][name] = []
                mse[dataset][targets][graphtype][name] = []

            mse[dataset][targets][graphtype]['deep'] = []
            r[dataset][targets][graphtype]['deep'] = []

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

    feature_importance = np.zeros((n_roi), dtype='float32')
    
    for i, graph_filename in enumerate(os.listdir(root_dir + graphtype)):

        ursi = graph_filename[4:13]
        ursis.append(ursi)

        try:
            y[i, :] = np.hstack((df1.loc[ursi, 'CCI'], df1.loc[ursi, :].iloc[4:].as_matrix()))

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

    x2 = np.zeros((n_subjects2, n_features), dtype='float32')
    y2 = np.zeros((n_subjects2, 1), dtype='float32')
    for i, graph_filename in enumerate(os.listdir(test_dir + graphtype)):

        ursi = graph_filename[4:13]
        # ursis.append(ursi)

        try:
            y2[i, 0] = df2.loc[ursi, 'CCI']

            graph_data = np.load(test_dir + graphtype + '/' + graph_filename)
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
            x2[i, :] = np.hstack((features, rich_hist, assortivity))

            # print('nans:', np.sum(np.isnan(graph_data)))
            # print(np.max(x[i, :]), np.min(x[i, :]), np.mean(x[i, :]))

        except KeyError as e:
            print(e)
            i -= 1

    # print(i, 'subjects')

    x_mms, y_mms = MinMaxScaler(), MinMaxScaler()
    x2_mms, y2_mms = MinMaxScaler(), MinMaxScaler()
    x = x_mms.fit_transform(x)
    y = y_mms.fit_transform(y)
    x2 = x2_mms.fit_transform(x2)
    y2 = y2_mms.fit_transform(y2)

    # x = normalize(x, axis=0)

    kf = KFold(n_splits=10)

    print('x:', x.shape)
    print('y:', y.shape)

    for k, (train_index, test_index) in enumerate(kf.split(range(x.shape[0]))):
        print('FOLD:', k+1)

        # print('nans:', np.sum(np.isnan(x_train)))
        # print('infs:', np.sum(np.isinf(x_train)))
        #
        # print('nans:', np.sum(np.isnan(y_train)))
        # print('infs:', np.sum(np.isinf(y_train)))

        n_targets = 1
        for targets in prediction_targets:
            print('targets:', targets)
            if 'CCI' in targets:
                n_targets = 1
                x_train = x[train_index]
                y_train = y[train_index][:, 0]
                x_test = x[test_index]
                y_test = y[test_index][:, 0]
            else:
                n_targets = n_outputs
                x_train = x[train_index]
                y_train = y[train_index]
                x_test = x[test_index]
                y_test = y[test_index]

            model = deep_mlp(n_features, n_targets)

            model_checkpoint = ModelCheckpoint(root_dir + 'best_model.hdf5', monitor="val_logcosh", verbose=0, save_best_only=True, save_weights_only=False, mode='min')
            lr_sched = lr_scheduler(model)

            adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
            model.compile(adam, 'logcosh', metrics=['mean_absolute_percentage_error', 'mean_squared_error', 'logcosh'])
            hist = model.fit(x_train, y_train, batch_size=100, epochs=1200, validation_split=0.1, callbacks=[model_checkpoint, lr_sched])

            train_losses.append(hist.history['loss'])
            val_losses.append(hist.history['val_loss'])
            model.load_weights(root_dir + 'best_model.hdf5')

            predictions = model.predict(x_test)
            actual = y_test

            r2 = r2_score(actual, predictions)
            error = mean_squared_error(actual, predictions)

            r['Templeton114'][targets][graphtype]['deep'].append(r2)
            mse['Templeton114'][targets][graphtype]['deep'].append(error)

            predictions = model.predict(x2)[:, 0]
            actual = y2

            r['Templeton255'][targets][graphtype]['deep'].append(r2_score(actual, predictions))
            mse['Templeton255'][targets][graphtype]['deep'].append(mean_squared_error(actual, predictions))

            print('deep', r2)
            for classifier, name in zip(classifiers, classifier_names):
                print('training', name)
                if isinstance(classifier, type(SVR())) and n_targets > 1:
                    r2s = []
                    errors = []

                    for n in range(n_targets):
                        classifier.fit(x_train, y_train[:, n])
                        predictions = classifier.predict(x_test)

                        r2s.append(classifier.score(x_test, y_test[:, n]))
                        errors.append(mean_squared_error(y_test[:, n], predictions))

                        if n == 0:
                            r2_2 = classifier.score(x2, y2)
                            predictions = classifier.predict(x2)

                            errors_2 = mean_squared_error(y2, predictions)

                    r2 = np.mean(r2s)
                    error = np.mean(errors)

                else:
                    classifier.fit(x_train, y_train)

                    r2 = classifier.score(x_test, y_test)
                    predictions = classifier.predict(x_test)
                    error = mean_squared_error(y_test, predictions)

                    predictions = classifier.predict(x2)
                    print(predictions.shape)
                    if n_targets > 1:
                        r2_2 = r2_score(y2, predictions[:, 0])
                        error_2 = mean_squared_error(y2, predictions[:, 0])
                    else:
                        r2_2 = r2_score(y2, predictions)
                        error_2 = mean_squared_error(y2, predictions)


                print(name, r2, error)
                r['Templeton114'][targets][graphtype][name].append(r2)
                mse['Templeton114'][targets][graphtype][name].append(error)

                r['Templeton255'][targets][graphtype][name].append(r2_2)
                mse['Templeton255'][targets][graphtype][name].append(error_2)

                if isinstance(classifier, type(RandomForestRegressor())):
                    feature_importance += classifier.feature_importances_[0:n_roi]

        plt.close()
        plt.plot(range(len(feature_importance)), feature_importance)
        plt.xlabel('ROI')
        plt.ylabel('Importance')
        plt.savefig(root_dir + '/results/' + graphtype + '_ROI_importance.png')

plt.close()


for targets in prediction_targets:
    for graphtype in graphtypes:
        print('Parcelation:', graphtype)
        for name in classifier_names + ['deep']:
            print(name, np.mean(r['Templeton114'][targets][graphtype][name]), np.std(r['Templeton114'][targets][graphtype][name]))
            print(name, np.mean(mse['Templeton255'][targets][graphtype][name]), np.std(mse['Templeton255'][targets][graphtype][name]))


# R-SQUARED BOXPLOT
plt.close()
for j, graphtype in enumerate(graphtypes):
    fig, ax = plt.subplots(2, 2, figsize=(32, 12))

    for i, targets in enumerate(prediction_targets):
        scores = []
        scores2 = []
        score_labels = []

        for name in classifier_names + ['deep']:
            scores.append(r['Templeton114'][targets][graphtype][name])
            scores2.append(r['Templeton255'][targets][graphtype][name])
            score_labels.append(name)

        bplot = ax[0, i].boxplot(scores, patch_artist=True, zorder=3)

        # ax[0, i].set_xticks(np.arange(1, len(scores)+1), score_labels)
        # ax[0, i].set_yticks(np.arange(0, 1.1, 0.1))
        ax[0, i].set_xlim(0, len(scores) + 1)
        ax[0, i].set_ylim(0, 1)

        colors = ['lightcoral', 'pink', 'hotpink', 'red', 'darkred', 'firebrick', 'm', 'darkblue']

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        ax[0, i].set_xlabel('Classifier')
        ax[0, i].set_ylabel('$r^2$')
        ax[1, i].set_xlabel('Classifier')
        ax[1, i].set_ylabel('$r^2$')

        bplot2 = ax[1, i].boxplot(scores2, patch_artist=True, zorder=3)

        # ax[1, i].set_xticks(np.arange(1, len(scores2)+1), score_labels)
        # ax[1, i].set_yticks(np.arange(0, 1.1, 0.1))
        ax[1, i].set_xlim(0, len(scores2) + 1)
        ax[1, i].set_ylim(0, 1)

        for patch, color in zip(bplot2['boxes'], colors):
            patch.set_facecolor(color)

        for k in [0, 1]:
            ax[k, i].xaxis.label.set_fontsize(24)
            ax[k, i].yaxis.label.set_fontsize(24)
            ax[k, i].yaxis.grid(True)
            for item in ([ax[k, i].title] + ax[k, i].get_xticklabels() + ax[k, i].get_yticklabels()):
                item.set_fontsize(20)

    ax[0, 0].set_title('All Measures')
    ax[0, 1].set_title('CCI Only')
    ax[0, 0].title.set_fontsize(32)
    ax[0, 1].title.set_fontsize(32)
    plt.setp(ax, xticks=np.arange(1, len(scores2)+1), xticklabels=score_labels, yticks=np.arange(0, 1.1, 0.1))

    plt.subplots_adjust()
    results_dir = root_dir + '/results/'
    plt.savefig(results_dir + graphtype + '_r2_boxplots.png')

plt.close()

# MEAN SQUARED ERROR BOXPLOT
for j, graphtype in enumerate(graphtypes):
    fig, ax = plt.subplots(2, 2, figsize=(32, 12))
    min = 100000
    max = -100000
    for i, targets in enumerate(prediction_targets):
        scores = []
        scores2 = []
        score_labels = []

        for name in classifier_names + ['deep']:
            scores.append(mse['Templeton114'][targets][graphtype][name])
            scores2.append(mse['Templeton255'][targets][graphtype][name])
            score_labels.append(name)

        bplot = ax[0, i].boxplot(scores, patch_artist=True, zorder=3)
        bplot2 = ax[1, i].boxplot(scores2, patch_artist=True, zorder=3)

        if np.min(scores) < min:
            min = np.min(scores)
        if np.max(scores) > max:
            max = np.max(scores)
        if np.min(scores2) < min:
            min = np.min(scores2)
        if np.max(scores2) > max:
            max = np.max(scores2)

        # ax[0, i].set_xticks(np.arange(1, len(scores)+1), score_labels)
        # ax[0, i].set_ylim(np.min(scores), np.max(scores))
        # ax[0, i].set_yticks(np.linspace(0, np.max(scores)*1.1, 5))
        ax[0, i].set_xlim(0, len(scores) + 1)
        ax[1, i].set_xlim(0, len(scores2) + 1)

        ax[0, i].set_xlabel('Classifier')
        ax[0, i].set_ylabel('Mean Squared Error')
        ax[1, i].set_xlabel('Classifier')
        ax[1, i].set_ylabel('Mean Squared Error')

        colors = ['lightcoral', 'pink', 'hotpink', 'red', 'darkred', 'firebrick', 'm', 'darkblue']

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # ax[1, i].set_xticks(np.arange(1, len(scores2)+1), score_labels)
        # ax[1, i].set_ylim(np.min(scores2), np.max(scores2))
        # ax[1, i].set_yticks(np.linspace(0, np.max(scores2)*1.1, 5))


        for patch, color in zip(bplot2['boxes'], colors):
            patch.set_facecolor(color)

        for k in [0, 1]:
            ax[k, i].xaxis.label.set_fontsize(24)
            ax[k, i].yaxis.label.set_fontsize(24)
            ax[k, i].yaxis.grid(True)
            for item in ([ax[k, i].title] + ax[k, i].get_xticklabels() + ax[k, i].get_yticklabels()):
                item.set_fontsize(20)

    ax[0, 0].set_title('All Measures')
    ax[0, 1].set_title('CCI Only')
    ax[0, 0].title.set_fontsize(32)
    ax[0, 1].title.set_fontsize(32)
    plt.setp(ax, xticks=np.arange(1, len(scores2)+1), xticklabels=score_labels, yticks=np.linspace(min, max, 10), ylim=(0, 0.5))

    plt.subplots_adjust()
    results_dir = root_dir + '/results/'
    plt.savefig(results_dir + graphtype + '_mse_boxplot.png')

plt.close()


plt.figure(figsize=(12, 6))
for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
    if i==0:
        plt.plot(train_loss, color='purple', label='Training')
        plt.plot(val_loss, color='darkred', label='Validation')
    else:
        plt.plot(train_loss, color='purple')
        plt.plot(val_loss, color='darkred')

    plt.legend(shadow=True, fontsize=20)
    plt.xlabel('Epoch Number', fontsize=20)
    plt.ylabel('Mean Squared Error (Validation)', fontsize=20)

plt.savefig(results_dir + graphtype + '_loss.png')

elapsed = start_time - time.time()

print(np.mod(elapsed, 60), 'minutes elapsed')