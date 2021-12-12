# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os


def get_update_df(conf):
    """
    From the direction of the dataset it clean it and return train and validation dataset for the training processes
    """
    df = pd.read_csv(conf['dir_df'])
    is_normal_df = df.groupby("image_id")["class_id"].agg(lambda s: (s == 14).sum()).reset_index().rename({
    "class_id": "target"}, axis=1)
    is_normal_df['target'] = (is_normal_df['target'] / 3).astype('int')

    skf  =  StratifiedKFold(n_splits = conf['n_splits'], random_state = 0, shuffle = True)
    folds = is_normal_df.copy()
    for f, (tr_idx, val_idx) in enumerate(skf.split(folds, folds.target)):
        folds.loc[val_idx, 'fold'] = int(f)
    folds['fold'] = folds['fold'].astype(int)  

    folds.image_id = folds.image_id + ".png"
    folds['path'] = [os.path.join(conf['DIR_TRAIN'], x) for x in folds.image_id]

    df_train = folds[folds['fold']!=conf['fold']]
    df_train = df_train.reset_index(drop=True)

    df_val = folds[folds['fold']==conf['fold']]
    df_val = df_val.reset_index(drop=True)

    return df_train, df_val

conf = {'dir_df':'', 'dir_df_test':'', 'DIR_TRAIN':'', 'DIR_TEST':'', 'dir_submision':'',
        'prob_mixup':0.8, 'num_classes':1, 'n_splits':5, 'fold':0,  
        'dir_best': '', 'dir_last':'', 'batch_size':8, 'num_epochs':15}