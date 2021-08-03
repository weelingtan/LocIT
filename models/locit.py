# -*- coding: UTF-8 -*-
"""

Full LocIT (with SSkNNO) algorithm.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# ----------------------------------------------------------------------------
# LocIT + SSkNNO
# ----------------------------------------------------------------------------

def apply_LocIT(Xs, Xt, ys=None, yt=None,
        psi=10, train_selection='random', scaling=True,
        k=10, supervision='loose'):
    """ Apply LocIT + SSkNNO.

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
    
    psi : int (default=10)
        Neighborhood size.
    
    train_selection : str (default='random')
        How to select the negative training instances:
        'farthest'  --> select the farthest instance
        'random'    --> random instance selected
        'edge'      --> select the (psi+1)'th instance
    
    scaling : bool (default=True)
        Scale the source and target domain before transfer.

    k : int (default=10)
        Number of nearest neighbors.
    
    supervision : str (default=loose)
        How to compute the supervised score component.
        'loose'     --> use all labeled instances in the set of nearest neighbors
        'strict'    --> use only instances that also count the instance among their neighbors
    
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
        Xs = scaler.fit_transform(Xs)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)

    # transfer
    ixt = _instance_transfer(Xs, Xt, ys, psi, train_selection)
    Xs_trans = Xs[ixt, :]
    ys_trans = ys[ixt]

    # combine
    X_combo = np.vstack((Xs_trans, Xt))
    y_combo = np.zeros(X_combo.shape[0], dtype=int)
    y_combo[:len(ys_trans)] = ys_trans

    # anomaly detection
    if ys.any():
        source_contamination = len(np.where(ys > 0)[0]) / len(np.where(ys != 0)[0])
    else:
        source_contamination = 0.1

    yt_scores = _ssknno_anomaly_detection(X_combo, y_combo, Xt,
        source_contamination, k, supervision)

    return yt_scores

# X = X_combo is (no. of transferred source samples + 550, 9) = (832, 9)
# y = y_combo is (no. of transferred source samples + 550, ) = (832,)
# Xt = Xt is (550, 9)
# c = source_contamination is percentage of anomalies in source data
# k = 10
# supervision = 'loose'
def _ssknno_anomaly_detection(X, y, Xt, c, k, supervision):
    """ Do the SSkNNO detection. """

    tol = 1e-10

    # construct the BallTree
    tree = BallTree(X, leaf_size=16, metric='euclidean')
    # Query the distance (D) of the k-nearest neighbours for all samples in X_combo
    # Note: D is a numpy array of shape (832, k+1) = (832, 11)
    # k=k+1 because if k = 1, the sample will just query itself as its nearest neighbour
    D, _ = tree.query(X, k=k+1)

    # compute gamma (scalar)
    # outlier_score is (832,)
    # It contains the distances of the k-th nearest neighbour for each of the X_combo samples
    # Hence, gamma = the 90th percentile value of the distances of the k-th nearest neighbour for each of the X_combo samples
    # gamma is used later in squashing function, hence, note that it squashes with the transferred source samples in X_combo also
    outlier_score = D[:, -1].flatten()
    gamma = np.percentile(outlier_score, int(100 * (1.0 - c))) + tol

    # labels and radii
    # labels is (832,)
    # radii is (832,)
    train_labels = y.copy()
    radii = D[:, -1].flatten() + tol

    # compute neighborhood for Xt samples
    # Now, we query only Xt samples, but using the tree that was fitted for X_combo
    # nn_radii is (550,), it contains the distances of the k-th nearest neighbour (from ALL OF X_combo) for EACH Xt
    # tree.query_radius will query for neighbors (from ALL OF X_combo) within a given radius for EACH Xt
    # Ixs_radius is (550,) each row shows the indices of the k nearest neighbours (from ALL OF X_combo) within the given radius for EACH Xt
    # Note: Each row can be different number of entries (but max no. is k=11), e.g:
    '''       
    array([293, 300, 305, 249,  12, 199, 208, 216, 471, 765, 323]),
    array([374, 386, 516, 294, 544, 640, 605, 694, 776, 788]),
    array([490, 638, 585, 464, 774, 295, 735, 702, 706, 412, 379]),
    '''
    # D_radius is (550,) each row shows the distances of the k nearest neighbours (from ALL OF X_combo) within the given radius for EACH Xt
    D, Ixs = tree.query(Xt, k=k+1, dualtree=True)
    nn_radii = D[:, -1].flatten()
    Ixs_radius, D_radius = tree.query_radius(
        Xt, r=nn_radii, return_distance=True, count_only=False)

    # compute prior (unsupervised component only)
    # prior is (550,), i.e. unsupervised component of the anomaly score for EACH Xt
    prior = _squashing_function(D[:, -1].flatten(), gamma)
    
    if not(y.any()):
        return prior

    # compute posterior (unsupervised + supervised components)
    # posterior is (550,) of zeros
    posterior = np.zeros(Xt.shape[0], dtype=float)
    for i in range(Xt.shape[0]):
        # Recall: Ixs_radius is (550,) each row shows the indices of the k nearest neighbours (from ALL OF X_combo) within the given radius for EACH Xt
        # Recall: D_radius is (550,) each row shows the distances of the k nearest neighbours (from ALL OF X_combo) within the given radius for EACH Xt
        # E.g: ndists is (11,) - distances, nixs is (11,) - indices, nn = 11
        ndists = D_radius[i].copy()
        nixs = Ixs_radius[i].copy()
        nn = len(ndists)

        # labels of the neighbors, weights
        # Recall: train_labels is y_combo is (832,) with 1.0 is anomaly, -1.0 is normal
        # Hence, labels is (11,), e.g: array([ 0,  0,  0,  0,  0, -1,  0, -1, -1, -1,  0]) where 0 means the neighbour is an unlabelled target sample, -1 means the neighbour is a labelled source sample, in this case -1 is normal
        # w is (11,), this is w(x_i, x_t) in the paper, Section 2
        labels = train_labels[nixs].copy()
        w = np.power(1.0 / (ndists + tol), 2)

        # supervised score component (GOAL: Find Ss)
        # E.g: ixl is array([5, 7, 8, 9])
        ixl = np.where(labels != 0.0)[0]
        
        # If there is at least one labelled source sample to calculate the supervised component, this will execute
        if len(ixl) > 0:
            # supervised score
            if supervision == 'loose':
                # Here, ixa is empty array([]) since for the labelled source samples ixl, there are no anomalies, all are normal samples
                # Hence, Ss = 0.0 (Note: np.sum(w[ixa]) sums up 1 if anomaly, 0 otherwise)
                # w[ixa] is the weights contributed by the LABELLED ANOMALOUS source samples
                # w[ixl] is the weights contributed by the LABELLED source samples
                # Hence, Ss is a scalar to indicate supervisory component of the anomaly score later (Note: Ss will be weighted by Ws later)
                ixa = np.where(labels > 0)[0]
                Ss = np.sum(w[ixa]) / np.sum(w[ixl])

            # weight of the supervised component (GOAL: Find Ws)
            # --> the number of labeled instances that also contain this instance as their neighbor
            # Recall: radii = D[:, -1].flatten() + tol is (832,), i.e. the distance of the furthest neighbour for each X_combo sample
           
            # Recall: Ixs_radius is (550,) each row shows the indices of the k nearest neighbours (from ALL OF X_combo) within the given radius for EACH Xt
            # Recall: D_radius is (550,) each row shows the distances of the k nearest neighbours (from ALL OF X_combo) within the given radius for EACH Xt
            # E.g: ndists is (11,) - distances, nixs is (11,) - indices, nn = 11
            # Recall: nixs = Ixs_radius[i].copy(), is the indices of the k nearest neighbours of this particular Xt sample

            # Thus, radii[nixs] is the distance of the the furthest neighbour for each of the k nearest neighbours of this Xt sample
            # ndists is the distances of the k nearest neighbours within the given radius for this Xt sample
            # Intuitively, we only want to consider the label of a transferred source instance if the source instance is similar to this Xt sample
            # Hence, we only consider instances in Xt's neighbourhood that also include Xt in their neighbourhood
            reverse_nn = np.where(ndists <= radii[nixs])[0]
            reverse_nn = np.intersect1d(ixl, reverse_nn) # Hence, reverse_nn are the samples that satisfy to be considered
            
            # Ws is W_l in the paper, but for this i-th Xt sample
            # Ws is the weight of the labeled instances in the neighbourhood of Xt samples (Ws will weight Ss later)
            # Recall: nn = len(ndists), i.e. total no. of k nearest neighbours of this i-th Xt sample
            # len(reverse_nn) is the no. of k nearest neighbours of this i-th Xt sample that satisfy to be considered
            Ws = len(reverse_nn) / nn

            # supervised score, new idea:
            if supervision == 'strict':
                if len(reverse_nn) > 0:
                    ixa = np.where(labels[reverse_nn] > 0)[0]
                    Ss = np.sum(w[ixa]) / np.sum(w[reverse_nn])
                else:
                    Ss = 0.0

        else:
            # supervised plays no role
            Ss = 0.0
            Ws = 0.0

        # combine supervised and unsupervised
        # Ws is W_l in the paper, but for this i-th Xt sample
        # Ws is the weight of the labeled instances in the neighbourhood of Xt samples
        # Ss is the supervised component of this i-th Xt sample, that considers nearby Xt's nearby labeled instances = distance weighted average of the labels
        # prior is the unsupervised component of this i-th Xt sample, uses KNNO algorithm that squashes the unsupervised anomaly scores
        posterior[i] = (1.0 - Ws) * prior[i] + Ws * Ss

    return posterior


def _instance_transfer(Xs, Xt, ys, psi, train_selection):
    """ Do the instance transfer. """

    tol = 1e-10

    # ns = 550, nt = 550
    ns, _ = Xs.shape
    nt, _ = Xt.shape

    # neighbor trees
    target_tree = BallTree(Xt, leaf_size=16, metric='euclidean')
    source_tree = BallTree(Xs, leaf_size=16, metric='euclidean')

    # 1. construct the transfer classifier
    # Query the indices (Ixs) of the k-nearest neighbours for all target domain samples Xt
    # Note: Ixs is a numpy array of shape (550, nt) = (550, 550)
    '''
    Ixsarray = 
    ([[  0, 398, 158, ..., 366, 249,  96],
       [  1, 388, 277, ...,  29, 249,  96],
       [  2,  95,  11, ...,  73, 249,  96],
       ...,
       [547, 539,  33, ...,   7, 229,  96],
       [548, 295, 385, ..., 366, 249,  96],
       [549, 187, 144, ..., 366, 249,  96]])
    '''

    _, Ixs = target_tree.query(Xt, k=nt)

    # X_train is (1100, 2)
    # y_train is (1100,)
    # random_ixs, after shuffle, is (550,) and contains indices from 0 to 549 randomly shuffled
    X_train = np.zeros((2 * nt, 2), dtype=np.float)
    y_train = np.zeros(2 * nt, dtype=np.float)
    random_ixs = np.arange(0, nt, 1)
    np.random.shuffle(random_ixs)

    # for i in range(550): (iterating over each Xt sample)
    for i in range(nt):
        # local mean and covaraiance matrix of the current point
        # Xt is (550, 9), psi = 20, hence Ixs[i, 1: psi+1] takes each row, 
        # i.e. the k-nearest neighbour indices for this current i-th Xt sample 
        # and then take the indices of the first 20 closest samples closest to this i-th Xt sample
        # NN_x is (20, 9), which is the 20 samples closest to this i-th Xt sample
        # mu_x is (9,), which is the mean of the features across all these 20 samples
        # C_x is (9, 9), which is the covariance matrix of these 9 features for NN_x (Note: must transpose NN_x first)
        NN_x = Xt[Ixs[i, 1:psi+1], :]
        mu_x = np.mean(NN_x, axis=0)
        C_x = np.cov(NN_x.T)

        # POS: local mean and covariance matrix of the nearest neighbor
        # nn_ix is the nearest neighbour for this current i-th Xt sample
        # Hence, over here, NN_nn, mu_nn and C_nn is analyzing the same local mean and covariance matrix,
        # BUT for the nearest neighbor of this i-th Xt sample instead
        # NN_nn is (20, 9), mu_nn is (9,), C_nn is (9, 9)
        nn_ix = Ixs[i, 1]
        NN_nn = Xt[Ixs[nn_ix, 1:psi+1], :]
        mu_nn = np.mean(NN_nn, axis=0)
        C_nn = np.cov(NN_nn.T)

        # NEG: local mean and covariance matrix of a randomly selected point
        # train_selection = 'farthest', hence we analyse for farthest neighbour for this current i-th Xt sample
        # NN_r is (20, 9), mu_r is (9,), C_r is (9, 9)
        if train_selection == 'random':
            r_ix = random_ixs[i]
        elif train_selection == 'edge':
            r_ix = Ixs[i, psi+2]
        elif train_selection == 'farthest':
            r_ix = Ixs[i, -1]
        else:
            raise ValueError(train_selection,
                'not valid!')
        NN_r = Xt[Ixs[r_ix, 1:psi], :]
        mu_r = np.mean(NN_r, axis=0)
        C_r = np.cov(NN_r.T)

        # training vectors
        # f_pos is (2,), this is the generated positive sample, for this i-th sample, for the SVM
        # f_neg is (2,), this is the generated negative sample, for this i-th sample, for the SVM
        f_pos = np.array([float(np.linalg.norm(mu_x - mu_nn)), float(
            np.linalg.norm(C_x - C_nn)) / float(np.linalg.norm(C_x) + tol)])
        f_neg = np.array([float(np.linalg.norm(mu_x - mu_r)), float(
            np.linalg.norm(C_x - C_r)) / float(np.linalg.norm(C_x) + tol)])

        # Recall, X_train is initialized as (1100, 2) of zeros, y_train is (1100,) of zeros
        # Positive examples (1.0) will be parked in X_train under even indices (i.e rows 0, 2, 4, ...)
        # Negative examples (0.0) will be parked in X_train under odd indices (i.e rows 1, 3, 5, ...)
        X_train[2*i, :] = f_pos
        y_train[2*i] = 1.0
        X_train[2*i+1, :] = f_neg
        y_train[2*i+1] = 0.0

    X_train = np.nan_to_num(X_train) # replace any Nan values with 0
    transfer_scaler = StandardScaler()
    X_scaled = transfer_scaler.fit_transform(X_train)
    
    # Train the SVM using these generated samples, X_train (1100, 2)
    clf = _optimal_transfer_classifier(X_scaled, y_train)

    # 2. determine which source instances to transfer
    # Query the indices (Ixs) of the k-nearest neighbours (in SOURCE domain) for all SOURCE domain samples Xs
    # Note: Ixs is a numpy array of shape (550, nt) = (550, 550)
    # Query the indices (Ixt) of the k-nearest neighbours (in TARGET domain) for all SOURCE domain samples Xs
    # Xs_feat is (550, 2) of zeros
    Xs_feat = np.zeros((ns, 2), dtype=np.float)
    _, Ixs = source_tree.query(Xs, k=psi+1)
    _, Ixt = target_tree.query(Xs, k=psi+1)
    
    # for i in range(550): (iterating over each Xs sample)
    for i in range(ns):
        # local mean and covariance matrix in the SOURCE domain transfer_ixs
        # Do the same for this i-th Xs sample
        NN_s = Xs[Ixs[i, 1:psi+1], :]  # nearest neighbors in the SOURCE domain
        mu_s = np.mean(NN_s, axis=0)
        C_s = np.cov(NN_s.T)

        # local mean and covariance matrix in the TARGET domain
        # Do the same for this i-th Xt sample
        NN_t = Xt[Ixt[i, :psi], :]  # nearest neighbors in the TARGET domain
        mu_t = np.mean(NN_t, axis=0)
        C_t = np.cov(NN_t.T)

        # f is (2,), the generated sample for this i-th source sample, to be classified by the trained SVM
        # Recall Xs_feat was initialized as (550, 2) of zeros
        f = np.array([float(np.linalg.norm(mu_s - mu_t)), float(
            np.linalg.norm(C_s - C_t)) / float(np.linalg.norm(C_s) + tol)])
        Xs_feat[i, :] = f

    Xs_feat = np.nan_to_num(Xs_feat)
    Xs_scaled = transfer_scaler.transform(Xs_feat)

    transfer_labels = clf.predict(Xs_scaled)

    # ixt is the indices of the source domain samples to be transferred
    ixt = np.where(transfer_labels == 1.0)[0]

    return ixt


# train = X_scaled is (1100, 2)
# labels = y_train is (1100, )
def _optimal_transfer_classifier(train, labels):
    """ optimal transfer classifier based on SVC """

    # tuning parameters
    # C is: Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
    # gamme is: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    # kernel is: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
    # In this case, we optimize hyperparameters over two types of kernels: 'linear' and 'rbf' with their respective hyperparameters
    tuned_parameters = [{'kernel': ['rbf'],
                        'C': [0.01, 0.1, 0.5, 1, 10, 100],
                        'gamma': [0.01, 0.1, 0.5, 1, 10, 100]},
                        {'kernel': ['linear'],
                        'C': [0.01, 0.1, 0.5, 1, 10, 100]}]
    
    # grid search
    # SVC(probability=True) means enabling probability estimates, must be enabled prior to calling fit
    svc = SVC(probability=True)
    clf = GridSearchCV(svc, tuned_parameters, cv=3, refit=True)
    clf.fit(train, labels)
    
    # return classifier
    return clf


def _squashing_function(x, p):
    """ Compute the value of x under squashing function with parameter p. """
    
    return 1.0 - np.exp(np.log(0.5) * np.power(x / p, 2))
