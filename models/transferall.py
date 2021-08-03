# -*- coding: UTF-8 -*-
"""

Transferall.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ----------------------------------------------------------------------------
# Transferall
# ----------------------------------------------------------------------------

# Xs is (550, 9)
# Xt is (550, 9)
# ys is (550,)
# yt is (550,)
def apply_transferall(Xs, Xt, ys=None, yt=None, scaling=True, k=10):
    """ Apply Transferall.

    Parameters
    ----------
    Xs : np.array of shape (n_samples, n_features), optional (default=None)
        The source instances.
    Xt : np.array of shape (n_samples, n_features), optional (default=None)
        The target instances.
    ys : np.array of shape (n_samples,), optional (default=None)
        The ground truth of the source instances.
    yt : np.array of shape (n_samples,), optional (default=None)
        The ground truth of the target instances.
    
    k : int (default=10)
        Number of nearest neighbors.

    Returns
    -------
    yt_scores : np.array of shape (n_samples,)
        Anomaly scores for the target instances.
    """

    # input
    # check_X_y is to check for consistent length, enforces X to be 2D and y to be 1D
    if ys is None:
        ys = np.zeros(Xs.shape[0])
    Xs, ys = check_X_y(Xs, ys)
    if yt is None:
        yt = np.zeros(Xt.shape[0])
    Xt, yt = check_X_y(Xt, yt)

    # StandardScaler() will normalize the features i.e. each column of X, INDIVIDUALLY 
    # So that each column/feature/variable will have μ = 0 and σ = 1.
    if scaling:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)

    # anomaly detection (kNN) - fit on source

    # X_combo is now (1100, 9)
    X_combo = np.vstack((Xs, Xt))
    
    # y_combo is now (1100,) with ys as the first 550 entries, the remaining 550 are zeros
    # Remaining 550 are assigned zeros because target domain labels are supposed to be unknown
    y_combo = np.zeros(X_combo.shape[0], dtype=int)
    y_combo[:len(ys)] = ys

    yt_scores = _kNN_anomaly_detection(X_combo, y_combo, Xt, k)
    
    return yt_scores


# X = X_combo is (1100, 9)
# y = y_combo is (1100,)
# Xt = Xt is (550, 9)
# k = k is 10
def _kNN_anomaly_detection(X, y, Xt, k):
    """ Apply kNN anomaly detection. """
    # ixl is (550,) numpy array containing only index positions of places where entries are NOT 0
    # Hence, here since all source domain samples are transferred, ixl is just the indices of all source samples
    ixl = np.where(y != 0)[0]
    # Xtr.shape is (550, 9), i.e. sieving out all the transferred source samples
    Xtr = X[ixl, :]
    # ytr.shape is (550,), i.e. sieving out all the transferred source samples
    ytr = y[ixl]
    
    # fit
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean', algorithm='ball_tree')
    clf.fit(Xtr, ytr)

    # predict
    # Here, we use the KNN classifier fitted on the transferred source samples to predict labels for target samples
    # yt_scores is (550, 2):
    '''
    [1. 0.]
    [0.6 0.4]
    [0. 1.]
    [0. 1.]
    [1. 0.]
    [0.7 0.3]
    [0.6 0.4]
    [0.4 0.6]
    [1. 0.]
    [0. 1.]
    ...
    '''
    yt_scores = clf.predict_proba(Xt)

    # clf.classes_ is array([-1,  1]), i.e. the class labels known to the classifier
    # Here, -1 is normal, 1 is anomaly
    if len(clf.classes_) > 1:
        # Since 1 is anomaly, ix is just telling us which position in clf.classes_ is the anomaly index
        # Hence, ix = 1 since 1 is in index 1 of array([-1, 1])
        ix = np.where(clf.classes_ == 1)[0][0]
        
        # Originally, yt_scores is (550, 2)
        # Now, yt_scores[:, ix] is (550,)
        # yt_scores = all the predicted probabilities (of an anomaly) from the KNN classifier for each sample Xt in target domain
        yt_scores = yt_scores[:, ix].flatten()
    else:
        yt_scores = yt_scores.flatten()

    # yt_scores is (550,) containing all the predicted probabilities (of an anomaly) from the KNN classifier for each sample Xt in target domain
    return yt_scores