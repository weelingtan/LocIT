# -*- coding: UTF-8 -*-
"""

Isolation Forest.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ----------------------------------------------------------------------------
# iForest
'''
Isolation Forest Algorithm.

Return the anomaly score of each sample using the IsolationForest algorithm

The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.

Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.
'''
# ----------------------------------------------------------------------------

# Xs is (550, 9)
# Xt is (550, 9)
# ys is (550,)
# yt is (550,)
def apply_iForest(Xs, Xt, ys=None, yt=None, scaling=True,
        n_estimators=100, contamination=0.1):
    """ Apply iForest.

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
    
    n_estimators : int (default=100)
        Number of estimators in the ensemble.

    contamination : float (default=0.1)
        The expected contamination (anomalies) in the data.

    Returns
    -------
    yt_scores : np.array of shape (n_samples,)
        Anomaly scores for the target instances.
    """

    # input
    if ys is None:
        ys = np.zeros(Xs.shape[0])
    Xs, ys = check_X_y(Xs, ys)
    if yt is None:
        yt = np.zeros(Xt.shape[0])
    Xt, yt = check_X_y(Xt, yt)

    # scaling
    if scaling:
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)

    # no transfer!

    # fit 
    clf = IsolationForest(n_estimators=n_estimators,
                          contamination=contamination,
                          n_jobs=1)
    clf.fit(Xt)

    # predict

    # Note: clf.decision_function(Xt) gives negative values as anomalies and non-negative ones as normals
    # Hence, we need to flip it here with * -1 since our dataset labels are 1 for anomaly, -1 for normal
    yt_scores = clf.decision_function(Xt) * -1
    # Standardize our predictions to be probabilities ranging from 0 to 1
    # Hence, yt_scores is (550,) = predicted probabilities (of an anomaly) from the Isolation Forest for each sample Xt in target domain
    yt_scores = (yt_scores - min(yt_scores)) / (max(yt_scores) - min(yt_scores))

    return yt_scores
