#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""

# Code source: Gael Varoqueux
#              Andreas Mueller
# Modified for Documentation merge by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import pylab as pl
import itertools as it
import colormap_util as cu
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

def preprocess(datasets, nshuffle=1, scale=False, test_size=0.4, max_samples=800):
    """
    preprocess dataset, split into training and test part

    """
    new_ds = []
    for ds in datasets:
        X, y = ds
        if scale:
            X = StandardScaler().fit_transform(X)
        if X.shape[0] > max_samples:
            train_size = int(max_samples * (1 - test_size))
            test_size_new = int(max_samples * test_size)
            print "Too many samples. Down sample to %d trainings and %d testings." % \
                    (train_size, test_size_new)
        else:
            train_size = 1 - test_size
            test_size_new = test_size
        for _ in range(0, nshuffle):
            X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, train_size=train_size, test_size=test_size_new)
            new_ds.append((X_train, X_test, y_train, y_test))
    return new_ds

def expand_2d(datasets):
    """
    Expand features in high dimension into set of 2D features

    """
    new_datasets = []
    indices = []
    for ds in datasets:
        X, y = ds
        for subset in it.combinations(range(0, X.shape[1]), 2):
            new_datasets.append((X[:, subset], y))
            indices.append(subset)
    return new_datasets, indices

def expand_range(x_min, x_max, margin=0.2):
    """
    Make expansion in 1-dimension of plotting area for better visualization.

    Args:
        x_min, x_max: lower and upper bound of current dimension
        margin: ratio of expansion to original size
    Returns:
        x_min, x_max: lower and upper bound after expansion
    """
    x_margin = (x_max - x_min) * margin
    x_min -= x_margin
    x_max += x_margin
    return x_min, x_max


def plot_classifiers(datasets, classifiers, data_names=None, clf_names=None,
                     margin=0.2, nxgrid=100, nygrid=100,
                     data_additional=True
                    ):
    if data_names is None:
        data_names = [ 'data' + str(i+1) for i in range(0, len(datasets))]
    if clf_names is None:
        clf_names = [ 'clf' + str(i+1) for i in range(0, len(classifiers))]

    subplot_nrow = len(datasets)
    subplot_ncol = len(classifiers)
    if data_additional:
        subplot_ncol += 1
    subplot_num = subplot_nrow * subplot_ncol
    while subplot_ncol < subplot_nrow:    # Place more subplots along horizental axis
        subplot_ncol += 2
        subplot_nrow = (subplot_num - 1) / subplot_ncol + 1

    figure = pl.figure()  #figsize=(27, 9))
    i = 1
    # iterate over datasets
    for dsname, ds in zip(data_names, datasets):
        X_train, X_test, y_train, y_test = ds

        nclasses = len(np.unique(np.r_[y_train, y_test]))
        x_min, x_max = expand_range(min(X_train[:, 0].min(), X_test[:, 0].min()),
                                    max(X_train[:, 0].max(), X_test[:, 0].max()),
                                    .2)
        y_min, y_max = expand_range(min(X_train[:, 1].min(), X_test[:, 1].min()),
                                    max(X_train[:, 1].max(), X_test[:, 1].max()),
                                    .2)
        xx, yy = np.mgrid[x_min:x_max:(nxgrid*1j), y_min:y_max:(nygrid*1j)]

        # just plot the dataset first
        cm = cu.get_colormap(nclasses)    #pl.cm.RdBu
        cm_bright = cu.get_colormap(nclasses)
        if data_additional:
            ax = pl.subplot(subplot_nrow, subplot_ncol, i)
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.4)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(dsname)
            i += 1

        # iterate over classifiers
        for name, clf in zip(clf_names, classifiers):
            ax = pl.subplot(subplot_nrow, subplot_ncol, i)
            clf.fit(X_train, y_train)
            score_train = clf.score(X_train, y_train)
            score_test = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            if nclasses == 2:
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            elif nclasses > 2:
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Put the result into a color plot
            if nclasses == 2:
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.6)
            elif nclasses > 2:
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.6)
                #ax.pcolormesh(xx, yy, Z, cmap=cm, alpha=.6)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.4)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name + ' : ' + dsname)
            ax.text(xx.max() - 0.12 * (xx.max() - xx.min()),
                    yy.min() + 0.06 * (yy.max() - yy.min()),
                    ('%.2f, %.2f' % (score_train, score_test)),
                    size=15,
                    horizontalalignment='right')
            i += 1

    figure.subplots_adjust(left=.02, right=.98)
    pl.show()

def plot_classifiers_roc(datasets, classifiers, data_names=None, clf_names=None,
                     margin=0.2, data_additional=True
                    ):
    if data_names is None:
        data_names = [ 'data' + str(i+1) for i in range(0, len(datasets))]
    if clf_names is None:
        clf_names = [ 'clf' + str(i+1) for i in range(0, len(classifiers))]

    subplot_nrow = len(datasets)
    subplot_ncol = len(classifiers)
    if data_additional:
        subplot_ncol += 1
    subplot_num = subplot_nrow * subplot_ncol
    while subplot_ncol < subplot_nrow:    # Place more subplots along horizental axis
        subplot_ncol += 2
        subplot_nrow = (subplot_num - 1) / subplot_ncol + 1
    while subplot_ncol > subplot_nrow * 4:   # don't place too many subplots along horizental axis
        subplot_ncol -= 2
        subplot_nrow = (subplot_num - 1) / subplot_ncol + 1

    figure = pl.figure()  #figsize=(27, 9))
    i = 1
    # iterate over datasets
    for dsname, ds in zip(data_names, datasets):
        X_train, X_test, y_train, y_test = ds

        nclasses = len(np.unique(np.r_[y_train, y_test]))
        if nclasses != 2:
            raise ValueError('Only binary classification supported')

        # just plot the dataset first
        cm = cu.get_colormap(nclasses)    #pl.cm.RdBu
        cm_bright = cu.get_colormap(nclasses)
        if data_additional:
            ax = pl.subplot(subplot_nrow, subplot_ncol, i)
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.4)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(dsname)
            i += 1

        # iterate over classifiers
        for name, clf in zip(clf_names, classifiers):
            ax = pl.subplot(subplot_nrow, subplot_ncol, i)
            clf.fit(X_train, y_train)

            if nclasses == 2:
                if hasattr(clf, "decision_function"):
                    proba = clf.decision_function(X_train)
                    proba_test = clf.decision_function(X_test)
                else:
                    proba = clf.predict_proba(X_train)[:, 1]
                    proba_test = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, thresh = roc_curve(y_train, proba)
            roc_auc = auc(fpr, tpr)
            fpr_test, tpr_test, thresh_test = roc_curve(y_test, proba_test)
            roc_auc_test = auc(fpr_test, tpr_test)

            ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            ax.plot(fpr_test, tpr_test, 'r', label='ROC curve (area = %0.2f)' % roc_auc)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_title(name + ':' + dsname)
            ax.text(0.9, 0.06,
                    ('%.2f, %.2f' % (roc_auc, roc_auc_test)),
                    size=15,
                    horizontalalignment='right')
            i += 1

    figure.subplots_adjust(left=.02, right=.98)
    pl.show()


if __name__ == '__main__':
    main()
