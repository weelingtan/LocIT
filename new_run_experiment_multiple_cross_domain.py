# -*- coding: UTF-8 -*-
"""

Run experiments.
: Tan Wee Ling (2021)
:credits: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

import sklearn as sk
import math
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

from torchvision import transforms
from torchvision import datasets
from ipywidgets import IntProgress

import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter

from collections import Iterable

import time
import datetime
import logging
import pickle
import json

import time, argparse

from sklearn.metrics import roc_auc_score

# transfer models
from models.locit import apply_LocIT
from models.transferall import apply_transferall
from models.coral import apply_CORAL
from models.damethod import apply_damethod
from models.damethod_v2 import apply_damethod_v2

# anomaly detection
from models.knno import apply_kNNO
from models.iforest import apply_iForest
# ----------------------------------------------------------------------------
# run experiment
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run transfer learning - anomaly detection experiment')
    parser.add_argument('-d', '--dataset', nargs='+', type=str, default='', help='dataset = folder in data/ directory')
    parser.add_argument('-m', '--method', nargs='+', type=str, default='', help='method to use')
    args, unknownargs = parser.parse_known_args()
    
    # difficulty dictionary
    transfer_difficulty = {
        'n1_a1': 1, # (n1, a1)
        'n1_a2': 2, # (n1, a2)
        'n2_a1': 4, # (n2, a1)
        'n12_a1': 3, # ((n1, n2), a1) -> not in shuttle and most other datasets
        'n12_a12': 3, # ((n1, n2), (a1, a2))
        'n2_a2': 5,
        'n2_n1': 6} # (n2, a2)
    # combos = [(n1, a1), (n1, a2), (n2, a1), (n2, a2), (n2, n1), ((n1, n2), (a1, a2))]
    # combo_names = ['n1_a1', 'n1_a2', 'n2_a1', 'n2_a2', 'n2_n1', 'n12_a12']

    # load the data
    # main_path is '/home/tanwl/LocIT'
    main_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(main_path, 'results')
    
    for dataset in args.dataset:
        handlers = [logging.FileHandler(dataset + '_damethod_v2_trial.log'), logging.StreamHandler()]
        format = '%(message)s'
        logging.basicConfig(level=logging.INFO, format=format, handlers=handlers)
        for method in tqdm(args.method):
            # data_path is '/home/tanwl/LocIT/data/shuttle'
            data_path = os.path.join(main_path, 'data_cross_domain', dataset)
            logging.info('The experiments are executed on the ' + dataset.lower() + ' data')
            logging.info('The experiments are executed using the ' + method.lower() + ' method')

            source_sets, target_sets = _load_and_preprocess_data(data_path)
            # source_sets is now a dictionary {'shuttle_source_n1_a1_v0': (550,10) Pandas Dataframe, 
            #                                   'shuttle_source_n1_a1_v1': (550,10) Pandas Dataframe, 
            #                                   ...}
            
            # target_sets is now a dictionary {'shuttle_v0': (550,10) Pandas Dataframe, 
            #                                   'shuttle_v1': (550,10) Pandas Dataframe, 
            #                                   ...}

            # apply algorithms - every combination of source and target
            auc_results = dict()
            dataset_name = ''

            # tgt_name is 'shuttle_v0'
            # target_data is (550, 10)
            for tgt_name, target_data in target_sets.items():
                start_time = time.time()
                # dataset_name is 'shuttle'
                dataset_name = tgt_name.split('_v')[0]

                # target data
                # Xt is (550, 9) numpy array i.e. the samples
                # yt is (550,) numpy array i.e. the labels
                Xt = target_data.iloc[:, :-1].values
                yt = target_data.iloc[:, -1].values
                
                # This is not used though
                # ixtl is (550,) numpy array containing only index positions of places where entries are NOT 0.0
                # e.g: testing = np.array([-1, -1, 0.0, -1, 0.0, -1, -1, -1, -1, 0.0, -1])
                # Hence, np.where(testing != 0.0)[0] gives array([ 0,  1,  3,  5,  6,  7,  8, 10])
                ixtl = np.where(yt != 0.0)[0]

                # This is not used though
                # nt is 550
                nt, _ = Xt.shape

                # transfer from each source domain

                # src_name is 'shuttle_source_n1_a1_v0', v1, ..., v4
                # ...
                # src_name is 'shuttle_source_n12_a12_v0', v1, .., v4
                # source_data is (550, 10)
                for src_name, source_data in source_sets.items():
                    
                    # source data
                    # Xs is (550, 9) numpy array i.e. the samples
                    # ys is (550,) numpy array i.e. the labels
                    Xs = source_data.iloc[:, :-1].values
                    ys = source_data.iloc[:, -1].values

                    # ns is 550
                    ns, _ = Xs.shape

                    # actual transfer + anomaly detection
                    # TRANSFER METHODS
                    # For all cases, target_scores is (550,), i.e. predicted probabilities of the sample being anomalous, for each of the 550 target domain samples
                    if method.lower() == 'locit':
                        target_scores = apply_LocIT(Xs, Xt.copy(), ys, yt.copy(),
                            k=10, psi=20, scaling=False, supervision='loose',
                            train_selection='farthest')

                    elif method.lower() == 'transferall':
                        target_scores = apply_transferall(Xs, Xt.copy(), ys, yt.copy(),
                            k=10, scaling=True)

                    elif method.lower() == 'coral':
                        target_scores = apply_CORAL(Xs, Xt.copy(), ys, yt.copy(),
                            scaling=True)

                    # UNSUPERVISED ANOMALY DETECTION METHODS
                    elif method.lower() == 'knno':
                        target_scores = apply_kNNO(Xs, Xt.copy(), ys, yt.copy(), scaling=False)

                    elif method.lower() == 'iforest':
                        target_scores = apply_iForest(Xs, Xt.copy(), ys, yt.copy(),
                            n_estimators=100, contamination=0.1)
                    
                    elif method.lower() == 'damethod':
                        auc, accuracy = apply_damethod(Xs, Xt.copy(), ys, yt.copy(),
                            dataset_name=dataset.lower(), src_name=src_name, tgt_name=tgt_name, 
                            batch_size=32, train_pct=0.8, epochs=50, device='cuda:0')
                    
                    elif method.lower() == 'damethod_v2':
                        auc = apply_damethod_v2(Xs, Xt.copy(), ys, yt.copy(),
                            dataset_name=dataset.lower(), src_name=src_name, tgt_name=tgt_name, 
                            batch_size=32, train_pct=0.8, epochs=50, device='cuda:0',
                            weighted_testing=True)
                    else:
                        raise ValueError(method,
                            'is not an implemented/accepted method')

                    # compute AUC
                    if method.lower() == 'damethod' or 'damethod_v2':
                        auc = auc
                    else:
                        auc = roc_auc_score(y_true=yt, y_score=target_scores)

                    # Transfer: 'shuttle_source_n1_a1_v0' --> 'shuttle_v0' AUC = auc
                    logging.info('Transfer:  ' + src_name + '\t-->\t' + tgt_name + '\tAUC = ' + str(auc))

                    # store the results
                    # sn = 'n1_a1'
                    sn = src_name.split('source_')[1].split('_v')[0]
                    if sn in auc_results.keys():
                        auc_results[sn].append(auc)
                    else:
                        auc_results[sn] = [auc]
                    # Hence, all auc scores for v0 to v4 for a particular transfer difficulty, e.g: n1_a1, will be stored in a list
                elapsed_time = time.time() - start_time
                logging.info('Done: ' + str(datetime.timedelta(seconds=round(elapsed_time))))
            # print results
            # AUC results on SHUTTLE
            # ----------------------
            # Difficulty level 1: auc_mean_of_v0_to_v4
            # ...
            # Done!
            logging.info('\n\nAUC results on {}'.format(dataset_name.upper()) + ' - ' + method.lower() + ':')
            logging.info('------------------'+'-'*(len(dataset_name) + len(method)))
            results_dict = {}
            for k, v in auc_results.items():
                logging.info('  Difficulty level {} ({}): \t{}'.format(transfer_difficulty[k], k, np.mean(v)))
                print(transfer_difficulty[k])
                print(np.mean(v))
                results_dict[transfer_difficulty[k]] = [np.mean(v)]
            
            results_df = pd.DataFrame(results_dict)
            results_df = results_df.rename(index={0: "AUC"})
            results_df.to_csv(os.path.join(results_dir, dataset + '_' + method + '.csv'))
    logging.info('\nDone!\n')


def _load_and_preprocess_data(data_path):

    """ Load and preprocess the data. """
    # src_path is '/home/tanwl/LocIT/data/shuttle/source'
    # tgt_path is '/home/tanwl/LocIT/data/shuttle/target'
    src_path = os.path.join(data_path, 'source')
    tgt_path = os.path.join(data_path, 'target')

    # source files is ['/home/tanwl/LocIT/data/shuttle/source/shuttle_source_n1_a1_v0.csv', ..., '/home/tanwl/LocIT/data/shuttle/source/shuttle_source_n12_a12_v4.csv']
    source_files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    source_files = [os.path.join(src_path, f) for f in source_files if '.csv' in f]

    # target files is ['/home/tanwl/LocIT/data/shuttle/target/shuttle_v0.csv', ... '/home/tanwl/LocIT/data/shuttle/target/shuttle_v9.csv']
    target_files = [f for f in os.listdir(tgt_path) if os.path.isfile(os.path.join(tgt_path, f))]
    target_files = [os.path.join(tgt_path, f) for f in target_files if '.csv' in f]

    # load the data
    source_sets = dict()
    for sf in source_files:
        # data is (550, 10) Pandas DataFrame that looks like this (shuffled already):
        # labels: 1.0 is anomaly, -1.0 is normal
        '''
                0    1     2    3     4     5     6     7     8  labels
        0    41.0 -4.0  86.0  0.0  42.0  15.0  46.0  45.0   0.0    -1.0
        1    43.0  0.0  84.0 -2.0  44.0  26.0  41.0  41.0   0.0    -1.0
        2    55.0  0.0  78.0  0.0  42.0  -2.0  23.0  37.0  14.0     1.0
        3    37.0  0.0  91.0  3.0   8.0   0.0  53.0  83.0  30.0    -1.0
        4    37.0  0.0  76.0 -7.0  28.0   0.0  39.0  47.0   8.0    -1.0
        ..    ...  ...   ...  ...   ...   ...   ...   ...   ...     ...
        545  43.0  0.0  85.0  0.0  42.0 -14.0  42.0  44.0   2.0    -1.0
        546  37.0  0.0  78.0  0.0  -4.0   4.0  42.0  83.0  42.0    -1.0
        547  43.0 -1.0  79.0  0.0  42.0 -15.0  35.0  37.0   2.0    -1.0
        548  56.0  0.0  76.0 -7.0  -4.0   0.0  20.0  81.0  62.0     1.0
        549  37.0  0.0  79.0  5.0  36.0 -15.0  42.0  43.0   2.0    -1.0
        '''
        data = pd.read_csv(sf, sep=',', index_col=0).sample(frac=1).reset_index(drop=True)
        # file_name is 'shuttle_source_n1_a1_v0'
        data.loc[(data['labels'] == -1.0), 'labels'] = 0.0
        file_name = os.path.split(sf)[1].split('.csv')[0]
        source_sets[file_name] = data
    # source_sets is now a dictionary {'shuttle_source_n1_a1_v0': (550,10) Pandas Dataframe, 
    #                                   'shuttle_source_n1_a1_v1': (550,10) Pandas Dataframe, 
    #                                   ...}

    target_sets = dict()
    for sf in target_files:
        data = pd.read_csv(sf, sep=',', index_col=0).sample(frac=1).reset_index(drop=True)
        data.loc[(data['labels'] == -1.0), 'labels'] = 0.0
        file_name = os.path.split(sf)[1].split('.csv')[0]
        target_sets[file_name] = data
    # target_sets is now a dictionary {'shuttle_v0': (550,10) Pandas Dataframe, 
    #                                   'shuttle_v1': (550,10) Pandas Dataframe, 
    #                                   ...}

    return source_sets, target_sets


if __name__ == '__main__':
    main()
    