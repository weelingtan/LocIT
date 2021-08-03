# -*- coding: UTF-8 -*-
"""

kNNo.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree


# ----------------------------------------------------------------------------
# kNNO
# ----------------------------------------------------------------------------

def apply_kNNO(Xs, Xt, ys=None, yt=None, scaling=True, k=10, contamination=0.1):
    """ Apply kNNO.
    k-distance is the distance of its k-th nearest neighbour in the dataset
    
    KNNO ranks all instances in a dataset by their k-distance, with higher distances signifying
    more anomalous instances
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

    contamination : float (default=0.1)
        The expected contamination in the data.

    Returns
    -------
    yt_scores : np.array of shape (n_samples,)
        Anomaly scores for the target instances.
    """

    # input
    if Xs is not None:
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
    tree = BallTree(Xt, leaf_size=16, metric='euclidean')

    # Query the distance (D) of the k-nearest neighbours for all target domain samples Xt
    # Note: D is a numpy array of shape (550, k+1) = (550, 11)
    # k=k+1 because if k = 1, the sample will just query itself as its nearest neighbour
    D, _ = tree.query(Xt, k=k+1)

    # predict
    # outlier_scores is (550,)
    # It contains the distances of the k-th nearest neighbour for each of the target domain samples
    outlier_scores = D[:, -1].flatten()

    # contamination = 0.1, hence int(100*(1-contamination)) = 90
    # Hence, gamma = the 90th percentile value of the distances of the k-th nearest neighbour for each of the target domain samples
    gamma = np.percentile(
        outlier_scores, int(100 * (1.0 - contamination))) + 1e-10

    # yt_scores is a squashed function of outlier_scores with parameter gamma
    # Hence, yt_scores is (550,) = predicted probabilities (of an anomaly) from the KNNO algorithm for each sample Xt in target domain
    # Higher values = more anomalous (monotonic increasing)
    yt_scores = _squashing_function(outlier_scores, gamma)
    
    return yt_scores

# Squashing function forces k-th nearest neighbour distances to be from 0 to 1
def _squashing_function(x, p):
    """ Compute the value of x under squashing function with parameter p. """
    
    return 1.0 - np.exp(np.log(0.5) * np.power(x / p, 2))
