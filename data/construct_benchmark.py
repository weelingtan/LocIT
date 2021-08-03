# -*- coding: UTF-8 -*-
"""

Construct benchmark for transfer learning for anomaly detection.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import pandas as pd
import sklearn as sk
import math, os, sys
import random
import itertools
import operator
from collections import Counter

from tqdm import tqdm

from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


# -------------
# VARIABLES
# -------------

INPUT_DIR = '/home/tanwl/LocIT/raw_datasets'
OUTPUT_DIR = '/Users/vincent/Benchmark_data/outlier_detection/set7/'

MIN_SIZE = 500
MIN_SUBSET = 50
A_PERCENT = 0.1

# dictionary with for each dataset: (N1, A1) method 5
CLASS_DICT_M5 = {
    'abalone': (2, 1),
    'covertype': (1, 0),
    'gas_sensors': (4, 5),
    'gesture_segmentation': (4, 2),
    'hand_posture': (4, 1),
    'landsat_satellite': (5, 3),
    'handwritten_digits': (1, 4),
    'letter_recognition': (1, 17),
    'pen_digits': (2, 1),
    'satimage': (5, 3),
    'segment': (2, 4),
    'sense_IT_acoustic': (2, 0),
    'sense_IT_seismic': (2, 1),
    'sensorless': (5, 3),
    'shuttle': (0, 2),
    'waveform': (2, 0),
    'poker': (0, 2) # Takes long computation time!
    }


# -------------
# MAIN
# -------------

# Counter({1: 37, 3: 6748, 0: 34108, 4: 2458, 2: 132, 6: 11, 5: 6}) USING TRAINING SET
# class 0 34108 (n1)
# class 2 132 (a1)
# class 3 6748 (n2)
# class 4 2458 (a2)

def main():
    # read each .csv file in the input directory
    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    csv_files = [f for f in files if '.csv' in f]
    # csv_files = ['shuttle.csv', 'abalone.csv', ..., etc.]
    # for each file construct the target sets
    for f in csv_files:
        print('')
        print('Processing file:', f)
        # name is 'shuttle'
        name = f.split('.')[0]

        # read the file and gather features, labels
        if name == 'poker':
            features, scaled_features, labels, ncl = load_data(os.path.join(INPUT_DIR, f), drop_classes=[7, 8, 9])
            # classes 0, 1 needs subsampling
            features, scaled_features, labels = subsample_data(features, scaled_features, labels, {0: 0.1, 1: 0.1})
        else:
            features, scaled_features, labels, ncl = load_data(os.path.join(INPUT_DIR, f))
        n_samples, n_dim = features.shape
        # n_samples = 14500 / 12332 (in this case, we subsampled)
        print('Number of datapoints:', n_samples)
        # n_dim = 9
        print('Number of features:', n_dim)
        # ncl = 7 / 5 (in this case, we subsampled)
        print('Number of classes:', ncl)
        

        # Counter({1: 37, 3: 6748, 0: 34108, 4: 2458, 2: 132, 6: 11, 5: 6}) USING TRAINING SET
        # class 0 34108 (n1)
        # class 2 132 (a1)
        # class 3 6748 (n2)
        # class 4 2458 (a2)

        # Counter({0: 11478, 3: 2155, 4: 809, 2: 39, 1: 13, 5: 4, 6: 2}) USING TEST SET
        # Counter({0: 11478, 2: 809, 1: 39, 4: 2, 3: 4}) SUBSAMPLED VERSION
        print('Datapoints per class:', Counter(labels))

        # get class combination from class dictionary
        
        # name = 'shuttle'/'poker'/'abalone'/etc. this will execute
        # For shuttle, previously_picked_classes = (0, 2)
        if name in CLASS_DICT_M5.keys():
            previously_picked_classes = CLASS_DICT_M5[name]
        # This will not execute
        else:
            previously_picked_classes = None

        # construct target sets: select the normal and anomaly class - normals and anomalies contain indices that span entire dataset!
        normals_ixs, anomalies_ixs, picked_classes = construct_target_sets_method_5(scaled_features, labels, ncl, n_samples, n_dim, previously_picked_classes)

        # pick source and target classes
        # possible_classes is {1, 3, 4, 5, 6} -> i.e. remaining classes
        # n1 is 0, a1 is 2
        # ncl is 7
        possible_classes = set([i for i in range(ncl)]) - set(picked_classes)
        n1, a1 = picked_classes[0], picked_classes[1]
        
        # This will not execute for shuttle
        if ncl == 3:
            assert len(possible_classes) == 1, 'Error - problem with amount of classes'
            # pick the one remaining class as n2
            n2 = random.sample(possible_classes, 1)[0]
            # class combos to sample datasets
            combos = [(n1, a1), (n2, a1), (n2, n1), ((n1, n2), a1)]  # 4 possible source domains
            combo_names = ['n1_a1', 'n2_a1', 'n2_n1', 'n12_a1']
        # This will execute for shuttle
        else:
            # select the largest of the remaining classes as n2
            # class_sizes is {3: 2155, 4: 809, 1: 13, 6: 2, 5: 4}
            # n2 is 3 (class 3)
            class_sizes = {k: v for k, v in Counter(labels).items() if k in possible_classes}
            n2 = max(class_sizes.items(), key=operator.itemgetter(1))[0]
            # select the remaining class or the largest of the remaining classes as a2
            # a2 is 4 (class 4)
            possible_classes = possible_classes - set([n2])
            class_sizes = {k: v for k, v in Counter(labels).items() if k in possible_classes}
            a2 = max(class_sizes.items(), key=operator.itemgetter(1))[0]
            # class combos to sample datasets
            # combos is [(0, 2), (0, 4), (3, 2), (3, 4), (3, 0), ((0, 3), (2, 4))]
            combos = [(n1, a1), (n1, a2), (n2, a1), (n2, a2), (n2, n1), ((n1, n2), (a1, a2))]  # 6 possible source domains
            combo_names = ['n1_a1', 'n1_a2', 'n2_a1', 'n2_a2', 'n2_n1', 'n12_a12']
        # Hence, we have now picked source and target classes
        # n1 = 0, a1 = 2, n2 = 3, a2 = 4


        # path to source and target set
        # OUTPUT_DIR = '/Users/vincent/Benchmark_data/outlier_detection/set7/'
        # name = 'shuttle'
        # Hence, '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10'
        # Hence, '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/target'
        # Hence, '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/source'
        if not os.path.exists(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100)))):
            os.makedirs(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))))
        if not os.path.exists(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'target')):
             os.makedirs(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'target'))
        if not os.path.exists(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'source')):
             os.makedirs(os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'source'))

        # define sampling n1 and a1 for source and target
        # sampling without replacement! Source and target instances cannot be equal!
        tgt_n1, src_n1 = {}, {}
        tgt_a1, src_a1 = {}, {}

        # Iterate through dictionaries of normals_ixs (class 0) and anomalies_ixs (class 2)
        # 0 is supervised AND unsupervised correct
        # 1 is unsupervised remaining
        # At the end, we have tgt_n1 = {0: array([  646, 10878,  5567, ...,  2170, 12134, 12939]),
        #                               1: array([], dtype=int64)}
        #                     src_n1 = {0: array([12382,  2766,  8393, ...,  7330,  9586,  3880]),
        #                               1: array([], dtype=int64)}
        #                     tgt_a1 = {0: array([], dtype=int64),
        #                               1: array([12861,  1445,  5058,  6543, 12740, 12545,  9448, 14149, 10470,
        #                                        8226,  2264,  4295, 14314,  4854,  7866,  3359])}
        #                      src_a1 = {0: array([], dtype=int64),
        #                                1: array([ 8235, 11565,  2701, 11088,  4014, 11427,  2658,  9712,  1105,
        #                                           7851,  1916,  4970, 13521,  3929,  1348,  8053, 13646])}
        # src_n1[0] and tgt_n1[0] are both (5738,) -> class 0 (normal) -> supervised AND unsupervised correct
        # src_n1[1] and tgt_n1[1] are both (0,) -> class 0 (normal) -> unsupervised remaining
        # src_al[0] and tgt_a1[0 are both (0,) -> class 2 (anomalies) -> supervised AND unsupervised correct
        # src_al[1] is (17,) and tgt_a1[1] is (16,) -> class 2 (anomalies) -> unsupervised remaining
        # tgt_XX is for target dataset
        # src_XX is for source dataset
        for i in [0, 1]:
            # normals, normals_ixs = index_set1 (class 0 - normal)
            ixs = normals_ixs[i].copy()
            random.shuffle(ixs)
            ln = len(ixs)
            tgt_n1[i] = ixs[:int(ln/2)]     # half is target, half is source (no overlap)
            src_n1[i] = ixs[int(ln/2):]

            # anomalies, anomalies_ixs = index_set2 (class 2 - anomalies)
            ixs = anomalies_ixs[i].copy()
            random.shuffle(ixs)
            la = len(ixs)
            tgt_a1[i] = ixs[:int(la/2)]     # half is target, half is source (no overlap)
            src_a1[i] = ixs[int(la/2):]

        # construct the target set(s) using tgt_n1 and tgt_a1
        # tgt_n1 is {0: array[...], 1: array[...]} for class 0 - normal
        # tgt_a1 is {0: array[...], 1: array[...]} for class 2 - anomalies
        # features is (14500, 9)
        # A_PERCENT is 0.1
        # dataset_path is '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/target'
        # name is 'shuttle'
        # versions = 10
        construct_data_sets(tgt_n1, tgt_a1, features, A_PERCENT, os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'target'), name, 10)

        # construct the source set(s) - construct multiple times to avoid lucky sampling
        # src_n1 is {0: array[...], 1: array[...]} for class 0 - normal
        # src_a1 is {0: array[...], 1: array[...]} for class 2 - anomalies
        # picked_classes is (0, 2)
        # combos is [(0, 2), (0, 4), (3, 2), (3, 4), (3, 0), ((0, 3), (2, 4))]
        # combo_names is ['n1_a1', 'n1_a2', 'n2_a1', 'n2_a2', 'n2_n1', 'n12_a12']
        # features is (14500, 9)
        # labels is (14500,)
        # A_PERCENT is 0.1
        # dataset_path is '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/source'
        # name is 'shuttle'
        # i is 0, 1, 2, 3, 4
        for i in range(5):
            construct_source_sets(src_n1, src_a1, picked_classes, combos, combo_names, features, labels, A_PERCENT, os.path.join(os.path.join(OUTPUT_DIR, name + '_a' + str(int(A_PERCENT * 100))), 'source'), name, i)


# -------------
# FUNCTIONS
# -------------

def construct_target_sets_method_4(features, labels, ncl, n_samples, n_dim, final_features, dataset_name=None):
    """ Construct the target sets: most confusion between classes. """

    class_combinations = np.unique(labels)

    normals = None
    anomalies = None

    # investigate each combination of classes
    confusion = np.inf
    miss_classified = 0
    best_combination = None
    for each in itertools.combinations(class_combinations, 2):
        # use precalculated or determined classes
        skip_rest = False
        if dataset_name in CLASS_DICT.keys():
            each = CLASS_DICT[dataset_name]
            skip_rest = True

        print('iteration with:', each)

        # select the classes
        idx_c1 = np.where(labels == each[0])[0]
        idx_c2 = np.where(labels == each[1])[0]
        idx = np.concatenate((idx_c1, idx_c2))
        selected_features = features[idx, :]
        selected_labels = labels[idx]

        # final features
        ff_class1 = final_features[idx_c1, :]
        ff_class2 = final_features[idx_c2, :]

        # cross-validation to separate classes
        class_probabilities = np.zeros(n_samples)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in skf.split(selected_features, selected_labels):
            # fit + predict
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
            clf.fit(selected_features[train_idx, :], selected_labels[train_idx])
            y_prob = clf.predict_proba(selected_features[test_idx, :])
            c = np.where(clf.classes_ == each[0])[0][0]
            class_probabilities[test_idx] = y_prob[:, c]

        # calculate confusion class 1
        idx1 = np.where(selected_labels == each[0])[0]
        cp = class_probabilities[idx1]
        idx_cc_class1 = np.where(cp >= 0.5)[0]
        new_miss_classified = len(idx1) - len(idx_cc_class1)
        new_confusion = np.sum(cp[idx_cc_class1])

        # calculate confusion class 2
        idx = np.where(selected_labels == each[1])[0]
        cp = 1.0 - class_probabilities[idx]
        idx_cc_class2 = np.where(cp > 0.5)[0]
        new_miss_classified += len(idx) - len(idx_cc_class2)
        new_confusion += np.sum(cp[idx_cc_class2])

        assert len(np.intersect1d(idx1, idx)) == 0, 'Error - normals and anomalies not off different class!'

        # adapt confusion etc.
        if new_confusion < confusion:
            # update the confusion metric
            confusion = new_confusion
            miss_classified = new_miss_classified
            best_combination = each

            # update the set of normals and anomalies
            # smallest class is the anomaly class
            if len(idx_cc_class1) > len(idx_cc_class2):
                normals = ff_class1[idx_cc_class1, :]
                anomalies = ff_class2[idx_cc_class2, :]
            else:
                normals = ff_class2[idx_cc_class2, :]
                anomalies = ff_class1[idx_cc_class1, :]

        if skip_rest:
            break

    # print classes picked
    print('Classes picked:', best_combination)

    return {0: normals, 1: np.array([])}, {0: anomalies, 1: np.array([])}

# features = scaled_features (14500, 9)
# labels = labels (14500,)
# ncl = 7
# n_samples = 14500
# n_dim = 9
# picked_classes = previously_picked_classes = (0,2)
def construct_target_sets_method_5(features, labels, ncl, n_samples, n_dim, picked_classes=None):
    """ Construct the target sets: most confusion between classes. """
    # class_combinations = array([0, 1, 2, 3, 4, 5, 6])
    class_combinations = np.unique(labels)

    normals = None
    anomalies = None

    # investigate each combination of classes
    skip_rest = False
    confusion = 0
    best_combination = None

    # Iterate over all combinations of class_combinations, total: 21 combinations
    # each will iterate over: (0, 1), (0, 2), ... (5, 6)
    for each in itertools.combinations(class_combinations, 2):
        # use precalculated or determined classes
        
        # This will execute since picked_classes = (0, 2)
        if picked_classes is not None:
            # each = (0, 2)
            each = picked_classes
            skip_rest = True

        print('iteration with:', each)

        # select the classes
        # ix shows the indices where the labels are 0 or 2, ix is (11517, )
        ix = np.where(np.isin(labels, each))[0]          # holds the indices of the currently studied classes
        # selected_features is (11517, 9)
        # selected_labels is (11517,)
        selected_features = features[ix, :].copy()
        selected_labels = labels[ix].copy()

        # cross-validation to separate classes
        # y_sup is (14500,) of zeros
        y_sup = np.zeros(n_samples)
        # KFold divides all the samples in k groups of sample (if k = n, then it is leave-one-out strategy)
        # Prediction is learned using k-1 folds, the fold left out is used for test
        # E.g: X = ["a", "b", "c", "d"], kf = KFold(n_splits=2), then 
        # for train, test in kf.split(X): print("%s %s" % (train, test))
        # gives [2 3] [0 1]
        #       [0 1] [2 3]

        # Now, StratifiedKFold is a variation of k-fold
        # It returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set
        # StratifiedKFold preserves the class ratios in both train and test dataset, good for imbalanced no. of classes

        # selected_labels contains: Counter({0: 11478, 2: 39})
        # At the end after prediction, y_sup contains: Counter({0.0: 14464, 2.0: 36})
        # Note: Only 11517 samples from selected_features/selecteD_labels were predicted for y_sup, the other zeros of y_sup are still zeros that were initialized
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in skf.split(selected_features, selected_labels):
            # fit + predict for y_sup
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5)
            clf.fit(selected_features[train_idx, :], selected_labels[train_idx])
            y_sup[test_idx] = clf.predict(selected_features[test_idx, :])

        # unsupervised classification using k-means clustering
        # y_unsup is final unsupervised prediction (class 0 or class 2) for each sample in selected_features (11517, 9)
        y_unsup = predict_unsupervised('kmeans', selected_features, selected_labels)

        # class index
        ix_c1 = np.where(selected_labels == each[0])[0]
        ix_c2 = np.where(selected_labels == each[1])[0]

        # supervised
        ix_ys_c1 = np.where(y_sup == each[0])[0]
        ix_ys_c2 = np.where(y_sup == each[1])[0]

        # unsupervised
        ix_yu_c1 = np.where(y_unsup == each[0])[0]
        ix_yu_c2 = np.where(y_unsup == each[1])[0]

        # supervised correct + remaining
        # ix_ys_c1_correct are the correctly predicted labels from supervised component for class 0
        # ix_ys_c2_correct are the correctly predicted labels from supervised component for class 2
        # ix_c1_remaining are the ground-truth labels that were not predicted, i.e. missed out, by supervised component for class 0
        # ix_c2_remaining are the ground-truth labels that were not predicted, i.e. missed out, by supervised component for class 2
        ix_ys_c1_correct = np.intersect1d(ix_c1, ix_ys_c1)
        ix_ys_c2_correct = np.intersect1d(ix_c2, ix_ys_c2)
        ix_c1_remaining = np.setdiff1d(ix_c1, ix_ys_c1_correct)
        ix_c2_remaining = np.setdiff1d(ix_c2, ix_ys_c2_correct)

        # unsupervised correct + remaining
        # ix_yu_c1_correct are correctly predicted labels from supervised AND unsupervised component for class 0
        # ix_yu_c1_faulty are are the ground-truth labels that were not predicted, i.e. missed out, by unsupervised component for class 0
        ix_yu_c1_correct = np.intersect1d(ix_ys_c1_correct, ix_yu_c1)
        ix_yu_c1_faulty = np.setdiff1d(ix_ys_c1_correct, ix_yu_c1_correct)
        ix_yu_c2_correct = np.intersect1d(ix_ys_c2_correct, ix_yu_c2)
        ix_yu_c2_faulty = np.setdiff1d(ix_ys_c2_correct, ix_yu_c2_correct)

        assert len(ix) == len(ix_c1) + len(ix_c2), 'Error - selected labels wrong'

        # store
        # We store: supervised AND unsupervised correct, unsupervised remaining, supervised remaining
        index_set1 = {0: ix[ix_yu_c1_correct], 1: ix[ix_yu_c1_faulty], 2: ix[ix_c1_remaining]}
        assert len(np.intersect1d(index_set1[0], index_set1[1])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set1[0], index_set1[2])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set1[1], index_set1[2])) == 0, 'Error in choosing class instances'
        assert np.sum([len(index_set1[i]) for i in range(3)]) == len(ix_c1), 'Error - choosing class instances'
        
        # We store: supervised AND unsupervised correct, unsupervised remaining, supervised remaining
        index_set2 = {0: ix[ix_yu_c2_correct], 1: ix[ix_yu_c2_faulty], 2: ix[ix_c2_remaining]}
        assert len(np.intersect1d(index_set2[0], index_set2[1])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set2[0], index_set2[2])) == 0, 'Error in choosing class instances'
        assert len(np.intersect1d(index_set2[1], index_set2[2])) == 0, 'Error in choosing class instances'
        assert np.sum([len(index_set2[i]) for i in range(3)]) == len(ix_c2), 'Error - choosing class instances'

        # confusion
        # len(index_set1[1]) = 0, len(index_set2[1]) = 35
        # combo_confusion = 35
        combo_confusion = len(index_set1[1]) + len(index_set2[1])

        # adapt confusion etc.
        # confusion = 0 initially on the first iteration, it is continuously updated
        
        # Only if current iteration's combo_confusion is high, then this will execute
        # Intuitively, if current iteration's combo_confusion is high, then it is hard to classify the target domain anomalies
        # This is ideal, so we should pick these two classes as normal vs anomalies since they are hard to classify
        if combo_confusion > confusion:
            # update the confusion metric
            confusion = combo_confusion

            # set the sets of normals and anomalies
            # smallest class is the anomaly class
            if len(index_set1[0]) > len(index_set2[0]):
                normals = index_set1
                anomalies = index_set2
                best_combination = each
            else:
                normals = index_set2
                anomalies = index_set1
                best_combination = (each[1], each[0])

        if skip_rest:
            break

    # print classes picked
    print('Classes picked:', best_combination, 'with', confusion, 'confusion')

    # normals = index_set1 = {0: ..., 1: ..., 2: ...}
    # anomalies = index_set2 = {0: ..., 1: ..., 2: ...}
    # best_combination = (0, 2)
    return normals, anomalies, best_combination


# construct the source set(s) - construct multiple times to avoid lucky sampling
# src_n1 = src_n1 is {0: array[...], 1: array[...]} for class 0 - normal
# src_a1 = src_a1 is {0: array[...], 1: array[...]} for class 2 - anomalies
# picked_classes is (0, 2)
# combos is [(0, 2), (0, 4), (3, 2), (3, 4), (3, 0), ((0, 3), (2, 4))]
# combo_names is ['n1_a1', 'n1_a2', 'n2_a1', 'n2_a2', 'n2_n1', 'n12_a12']
# features is (14500, 9)
# labels is (14500,)
# anomaly_percent = A_PERCENT is 0.1
# dataset_path is '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/source'
# name is 'shuttle'
# version is 0, 1, 2, 3, 4
def construct_source_sets(src_n1, src_a1, picked_classes, combos, combo_names, features, labels, anomaly_percent, dataset_path, name, version):
    """ Construct the source sets. """

    # construct the source domains
    # ii iterates from 0, 1, 2, 3, 4, 5
    # each iterates from (0, 2), (0, 4), (3, 2), (3, 4), (3, 0), ((0, 3), (2, 4))
    for ii, each in enumerate(combos):
        print('Source combination: ', combo_names[ii])
        # normals
        n1 = each[0] # n1 = 0
        n2 = None

        # This will only execute for each = ((0, 3), (2, 4))
        if type(n1) == tuple:
            n1 = each[0][0] # n1 = 0
            n2 = each[0][1] # n2 = 3
        
        # This will execute for n1 = 0
        if n1 == picked_classes[0]:
            # idx is (5739,) combining source normals supervised AND unsupervised together with unsupervised remaining
            idx = np.concatenate((src_n1[0], src_n1[1]))
            # normals1 is (5739, 9)
            normals1 = features[idx, :]
        # This will execute for n1 =/= 0
        else:
            idx = np.where(labels == n1)[0]
            normals1 = features[idx, :]
        
        # This will only execute for each = ((0, 3), (2, 4))
        if n2 is not None:
            # n2 = 3, this will not execute
            if n2 == picked_classes[0]:
                idx = np.concatenate((src_n1[0], src_n1[1]))
                normals2 = features[idx, :]
            # This will execute
            else:
                # idx is (39,)
                idx = np.where(labels == n2)[0]
                # normals2 is (39, 9)
                normals2 = features[idx, :]
        # This will execute for each =/= ((0, 3), (2, 4))
        else:
            normals2 = np.array([])

        # anomalies
        a1 = each[1] # a1 = 2
        a2 = None
        
        # This will only execute for each = ((0, 3), (2, 4))
        if type(a1) == tuple:
            a1 = each[1][0] # a1 = 2
            a2 = each[1][1] # a2 = 4

        # This will execute for a1 = 2
        if a1 == picked_classes[1]:
            # idx is (17,)
            idx = np.concatenate((src_a1[0], src_a1[1]))
            # anomalies1 is (17, 9)
            anomalies1 = features[idx, :]
        # This will execute for a1 =/= 2
        else:
            idx = np.where(labels == a1)[0]
            anomalies1 = features[idx, :]
        
        # This will only execute for each = ((0, 3), (2, 4))
        if a2 is not None:
            # a2 = 4, this will not execute
            if a2 == picked_classes[1]:
                idx = np.concatenate((src_a1[0], src_a1[1]))
                anomalies2 = features[idx, :]
            # This will execute
            else:
                # idx is (2155,)
                idx = np.where(labels == a2)[0]
                # anomalies2 is (2155, 9)
                anomalies2 = features[idx, :]
        # This will execute for each =/= ((0, 3), (2, 4))
        else:
            anomalies2 = np.array([])

        # select how many instances
        # n1 = 5739, n2 = 0, a1 = 17, a2 = 0
        # pn2 = 0, pn1 = 500
        n1, n2 = len(normals1), len(normals2)
        a1, a2 = len(anomalies1), len(anomalies2)
        pn2 = min(n2, int(0.5 * MIN_SIZE))
        pn1 = min(n1, MIN_SIZE - pn2)

        # a_size = 50
        # pa2 = 0, pa1 = 17
        a_size = int(math.ceil(anomaly_percent * (pn1 + pn2)))
        pa2 = min(a2, int(0.5 * a_size))
        pa1 = min(a1, a_size - pa2)
        
        # normals: 500 / (5739, 9) - 0 / (0,) - 17 / (17, 9) - 0 / (0,)
        print('normals:', pn1, '/', normals1.shape, '-', pn2, '/', normals2.shape, '-', \
              pa1, '/', anomalies1.shape, '-', pa2, '/', anomalies2.shape)

        # select the instances
        # fnorm1 is (500, 9)
        idx1 = np.random.choice(n1, pn1, replace=False)
        fnorm1 = normals1[idx1, :]

        # This will not execute since pn2 = 0
        if pn2 > 0:
            idx2 = np.random.choice(n2, pn2, replace=False)
            fnorm2 = normals2[idx2, :]
            nn = np.vstack((fnorm1, fnorm2))
        # This will execute
        else:
            nn = fnorm1

        # fanom1 is (17, 9)
        idx1 = np.random.choice(a1, pa1, replace=False)
        fanom1 = anomalies1[idx1, :]

        # This will not execute since pa2 = 0
        if pa2 > 0:
            idx2 = np.random.choice(a2, pa2, replace=False)
            fanom2 = anomalies2[idx2, :]
            aa = np.vstack((fanom1, fanom2))
        # This will execute
        else:
            aa = fanom1

        # combine data and labels
        # nn are normals -> given -1 as labels
        # aa are anomalies -> given 1 as labels
        data = np.vstack((nn, aa))
        temp_labels = np.ones(len(data)) * -1
        temp_labels[-len(aa):] = 1

        # store results
        # Each df is (516, 10), with last column as labels (-1 -1, ..., 1, 1), normals first then anomalies
        df = pd.DataFrame(data)
        df['labels'] = temp_labels
        df.to_csv(os.path.join(dataset_path, name + '_source_' + combo_names[ii] + '_v' + str(version) + '.csv'), sep=',')
        # '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/source/shuttle_source_n1_a1_v0.csv, shuttle_source_n1_a1_v1.csv, ..., shuttle_source_n1_a1_v4.csv'
        # '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/source/shuttle_source_n1_a2_v0.csv, shuttle_source_n1_a2_v1.csv, ..., shuttle_source_n1_a2_v4.csv'
        # ...
        # '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/source/shuttle_source_n12_a12_v0.csv, shuttle_source_n12_a12_v1.csv, ..., shuttle_source_n12_a12_v4.csv'
        # Altogether, there are 5 (n1_a1, ..., n12_a12) * 5 (v0, ..., v4) = 25 csv files


# construct the target set(s)
# norm = tgt_n1 is {0: array[...], 1: array[...]} for class 0 - normal
# anom = tgt_a1 is {0: array[...], 1: array[...]} for class 2 - anomalies
# features is (14500, 9)
# anomaly_percent = A_PERCENT is 0.1
# dataset_path is '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/target'
# name is 'shuttle'
# versions = 10
def construct_data_sets(norm, anom, features, anomaly_percent, dataset_path, name, versions):
    """ Construct and store the datasets. """

    # n = 5738 (class 0), a = 16 (class 2)
    # n0 = 5738 (normal, supervised AND unsupervised correct), n1 = 0 (normal, unsupervised remaining)
    # a0 = 0 (anomaly, supervised AND unsupervised correct), a1 = 16 (anomaly, unsupervised remaining)
    n, a = len(norm[0]) + len(norm[1]), len(anom[0]) + len(anom[1])
    n0, n1 = len(norm[0]), len(norm[1])
    a0, a1 = len(anom[0]), len(anom[1])

    # normals (try sampling equally from each set)
    # n0 is normal, supervised AND unsupervised correct = 5738
    # n1 is normal, unsupervised remaining = 0
    # MIN_SUBSET = 50
    # MIN_SIZE = 500
    # pn1 = 0, pn0 = 500
    pn1 = min(n1, max(MIN_SUBSET, int(n1 / n * MIN_SIZE))) # min(0, max(50, 0 / 5738 * 50))
    pn0 = min(n0, MIN_SIZE - pn1) # min(5738, 500 - 0)

    # anomalies
    # a_size = 50
    # a0 is anomaly, supervised AND unsupervised correct = 0
    # a1 is anomaly, unsupervised remaining = 16
    # anomaly_percent = 0.1
    # pa1 = 16, pa0 = 0
    a_size = int(int(math.ceil(anomaly_percent * (pn1 + pn0)))) # int(int(math.ceil(0.1 * (0 + 500)))) = 50
    pa1 = min(a1, max(int(a_size / 2), int(a1 / a * a_size))) # min(16, max(int(50 / 2), int(16 / 16 * 50)))
    pa0 = min(a0, a_size - pa1)

    # good normals: 500 / 5738 - bad normals: 0 / 0
    # good anomalies: 0 / 0 - bad anomalies: 16 / 16
    print('good normals:', pn0, '/', n0, '- bad normals:', pn1, '/', n1)
    print('good anomalies:', pa0, '/', a0, '- bad anomalies:', pa1, '/', a1)

    # versions = 10, i.e. 10 different random permutations of samples drawn
    for i in range(versions):
        # normals
        # This will execute
        if n0 > 0:
            # n0 = 5738, pn0 = 500
            ix = norm[0][np.random.choice(n0, pn0, replace=False)] # choose pn0 out of n0 samples
            normals0 = features[ix, :]
        # This will not execute
        else:
            normals0 = np.array([])
        
        # This will not execute
        if n1 > 0:
            ix = norm[1][np.random.choice(n1, pn1, replace=False)] # choose pn1 out of n1 samples
            normals1 = features[ix, :]
        # This will execute
        else:
            normals1 = np.array([])

        # anomalies
        if a0 > 0:
            ix = anom[0][np.random.choice(a0, pa0, replace=False)] # choose pa0 out of a0 samples
            anomalies0 = features[ix, :]
        # This will execute
        else:
            anomalies0 = np.array([])
        # This will execute
        # a1 = 16, pa1 = 16
        if a1 > 0:
            ix = anom[1][np.random.choice(a1, pa1, replace=False)] # choose pa1 out of a1 samples
            anomalies1 = features[ix, :]
        # This will not execute
        else:
            anomalies1 = np.array([])

        # As you can see, no distinction between n0 and n1 / a0 and a1 in the end, all are combined together
        # combine "good normals" and "bad normals"
        if len(normals1) > 0:
            nn = np.vstack((normals0, normals1))
        else:
            nn = normals0
        # combine "good anomalies" and "bad anomalies"
        if len(anomalies1) > 0:
            aa = np.vstack((anomalies0, anomalies1))
        else:
            aa = anomalies0

        # combine data and labels
        # nn are normals -> given -1 as labels
        # aa are anomalies -> given 1 as labels
        data = np.vstack((nn, aa))
        labels = np.ones(len(data)) * -1
        labels[-len(aa):] = 1

        # store results
        # Each df is (516, 10), with last column as labels (-1 -1, ..., 1, 1), normals first then anomalies
        df = pd.DataFrame(data)
        df['labels'] = labels
        df.to_csv(os.path.join(dataset_path, name + '_v' + str(i) + '.csv'), sep=',')
        # '/Users/vincent/Benchmark_data/outlier_detection/set7/shuttle_a10/target/shuttle_v0.csv, shuttle_v1.csv, ..., shuttle_v9.csv


# method = 'kmeans'
# features = selected_features, (11517, 9) with classes 0 and 2 only
# labels = selected_labels (11517,) with classes 0 and 2 only
def predict_unsupervised(method, features, labels):
    """ Predict the class of the instances. """

    if method == 'kmeans':
        # elbow method to select number of clusters
        mean_dist = []
        for k in [5, 10, 15, 20, 25, 30]:
            # KMeans is clustering, with k clusters
            # inertia = the cluster's sum of squares ( sum: || x_i - miu_j||^2 ) where miu_j is cluster's mean
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(features)
            mean_dist.append(kmeans.inertia_ / features.shape[0])

        # Elbow: explain x % of the variance
        # mean_dist = [1453.3034359525423, 591.2673981351678, 383.40473569900405,278.82278932595017, 225.88687018707014, 178.1071782722973]
        # np.diff(mean_dist) = array([-862.03603782, -207.86266244, -104.58194637,  -52.93591914, -47.77969191])
        # variance = array([862.03603782, 207.86266244, 104.58194637,  52.93591914, 47.77969191])
        # distortion_percent = array([0.67600264, 0.83900709, 0.92101952, 0.9625315 , 1.        ])
        # idx = array([3, 4])
        # best_k = 20
        explained_var = 0.95
        variance = np.diff(mean_dist) * -1
        distortion_percent = np.cumsum(variance) / (mean_dist[0] - mean_dist[-1])
        idx = np.where(distortion_percent > explained_var)[0]
        best_k = (idx[0] + 1) * 5

        # kmeans clustering
        # Note: y_pred is (11517,) but has labels 0, 1, 2, ..., 17, 18, 19 (Cluster Labels)
        kmeans = KMeans(n_clusters=best_k)
        kmeans.fit(features)
        y_pred = kmeans.predict(features)  # cluster labels

    elif method == 'gmm':
        pass

    # determine the class label of each cluster
    # l1 = 0, l2 = 2
    # class_1 and class_2 are (20,) of zeros
    l1, l2 = np.unique(labels)
    class_1 = np.zeros(best_k)
    class_2 = np.zeros(best_k)

    # i runs from 0 to 11517
    # cl is the class label (0, 1, ..., or 19)
    for i, cl in enumerate(y_pred):
        # Note: selected_labels only contains 0s and 2s
        # If this i-th entry in selected_labels equals 0, then class_1[cl] plus one
        if labels[i] == l1:
            class_1[cl] += 1
        # Else if this i-th entry in selected_labels equals 2, then class_2[cl] plus one
        else:
            class_2[cl] += 1
    # class_1 is array([2400, 2, 1, 1, 1, 889, 740, 5, 401, 2, 1, 1202, 714, 3, 3152, 1234, 722, 3, 4, 1])
    # class_2 is array([ 8.,  0.,  0.,  0.,  0.,  8., 12.,  0.,  0.,  0.,  0.,  1.,  1., 0.,  4.,  0.,  5.,  0.,  0.,  0.])
    class_dict = dict()
    for i in range(best_k):
        if class_1[i] > class_2[i]:
            class_dict[i] = l1
        else:
            class_dict[i] = l2

    # class_dict is {0: 0, 1: 0, 2: 0, 3: 0, ..., 19: 0}, i.e. dictonary with keys: each cluster label, values: class 0 or 2
    # predict the labels of individual instances
    # y_pred_final is (11517,) of zeros
    # Note: y_pred is (11517,) but has labels 0, 1, 2, ..., 17, 18, 19 (Cluster Labels)
    y_pred_final = np.zeros(len(labels))
    for i, l in enumerate(y_pred):
        y_pred_final[i] = class_dict[l]

    # y_pred_final returns final unsupervised prediction (class 0 or class 2) for each sample in selected_features (11517, 9)
    return y_pred_final

# Refer to shuttle preprocessing Jupyter Notebook for details
# dataset.csv must start in the format as shown in shuttle_source_n1_a1_v0.csv, with comma separating values
# Also, labels column must be named 'labels'. 
# To achieve this via raw shuttle.tst file from UCI repo:
# data = pd.read_csv('shuttle.tst', sep=' ',header=None) 
# Then, data.to_csv('shuttle_tst.csv', sep=',')
# Do this standardization across all the different raw datasets first before implementing load_data

# return features, scaled_features, labels (labels are converted to start from 0), ncl

# features is (14500, 9), unscaled
# scaled_features is (14500, 9), scaled
# labels is (14500,), running from 0, 1, 2, 3, 4, 5, 6 (recall original number of classes was 7)
# ncl = 7 (scalar)

# SUBSAMPLED VERSION
# features is (12332, 9), unscaled
# scaled_features is (12332, 9), scaled
# labels is (12332,), running from 0, 1, 2, 3, 4 (recall original number of classes was 7, we dropped 2 classes)
# ncl = 5 (scalar)
def load_data(dataset_path, drop_classes=[]):
    """ Load data from file. """
    data = pd.read_csv(dataset_path, sep=',').iloc[:, 1:]
    features = data.iloc[:, :-1].values.astype(float)
    labels = data.iloc[:, -1].values
    for c in drop_classes:
        idx = np.where(labels != c)[0]
        features = features[idx, :]
        labels = labels[idx]

    # z-normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # Rename our labels from 1, 2, ..., 7 to become 0, 1, ... , 6
    classes_ = np.unique(labels)
    class_dict = dict()
    for i, c in enumerate(classes_):
        class_dict[c] = i
    ncl = i + 1

    # change the label vector to become 0, 1, ..., 6 instead of 1, 2, ..., 7
    labels = np.array([class_dict[l] for l in labels])
    classes_ = np.array([i for i in range(ncl)])

    return features, scaled_features, labels, ncl


# features is (14500, 9), unscaled
# scaled_features is (14500, 9), scaled
# labels is (14500,), running from 0, 1, 2, 3, 4, 5, 6 (recall original number of classes was 7)
# ncl = 7 (scalar)

# SUBSAMPLED VERSION
# features is (12332, 9), unscaled
# scaled_features is (12332, 9), scaled
# labels is (12332,), running from 0, 1, 2, 3, 4 (recall original number of classes was 7, we dropped 2 classes)
# subsamp = {0 : 0.1, 1: 0.1}
def subsample_data(features, scaled_features, labels, subsamp): # This is only for poker dataset
    """ Subsample the data. """

    # k is class, will iterate from class 0 to class 1
    # v is fraction to sample, i.e. 0.1, sample 10% of the current class being iterated
    for k, v in subsamp.items():
        ix = np.where(labels == k)[0]
        ix_rest = np.where(labels != k)[0]
        sample_ix = np.random.choice(ix, int(v * len(ix)), replace=False)
        keep_ix = np.union1d(ix_rest, sample_ix)
        # subsample
        features = features[keep_ix, :]
        scaled_features = scaled_features[keep_ix, :]
        labels = labels[keep_ix]

    return features, scaled_features, labels


# -------------
# RUN SCRIPT
# -------------

if __name__ == '__main__':
    sys.exit(main())
