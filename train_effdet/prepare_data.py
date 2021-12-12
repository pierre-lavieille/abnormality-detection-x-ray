# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold



def update_bboxes(df, img_size):
    """
    Take the row dataframe of the doctors, rezised it and merge the close boxes

    Args:
        df ([type]): The observation dataframe
        img_size ([type]): Final image size you want

    """
    data = df.copy()
    data.loc[data["class_id"] != 14, ["x_min"]] = (data['x_min'] / data['width'])
    data.loc[data["class_id"] != 14, ["y_min"]] = (data['y_min'] / data['height'])
    data.loc[data["class_id"] != 14, ["x_max"]] = (data['x_max'] / data['width'])
    data.loc[data["class_id"] != 14, ["y_max"]] = (data['y_max'] / data['height'])
    
    image_ids = data["image_id"].unique()
    list_image = []
    list_boxes = []
    list_cls = []
    for image_id in tqdm(image_ids):
        records = data[data['image_id']==image_id].reset_index(drop=True)
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
        scores = [1.0]*len(boxes)
        labels = [float(i) for i in records['class_id'].values]
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels],weights=None,iou_thr=0.4,
                                          skip_box_thr=0.0001)
        list_image.extend([image_id]*len(boxes))
        list_boxes.extend(boxes)
        list_cls.extend(labels.tolist())
        
    data2 = pd.DataFrame(list(zip(list_image, list_cls)), 
                      columns=['image_id', 'class_id'])
    data2['x_min'], data2['y_min'], \
    data2['x_max'], data2['y_max'] = np.transpose(list_boxes)
    data2[['x_min', 'y_min', 'x_max', 'y_max']] = (img_size*data2[['x_min', 'y_min', 'x_max',
                                                                'y_max']]).astype(int)
    
    data = data.drop_duplicates(subset ="image_id")
    metadata = ['image_id', 'width', 'height', 'PatientSex', 'PatientWeight', 
                'PatientAge','PhotometricInterpretation', 'RescaleSlope', 'RescaleIntercept']
    data2 = data2.merge(data[metadata], on='image_id', how='left')
    data2['class_id'] = data2['class_id'].astype('int32')
    
    return data2

def k_fold_split(dataset, conf, shuffle=True, random_state=None):
    """
    Generate a specific splitting so that each split as the same number of abnormality

    Args:
        dataset (dataframe): the dataset
        conf (dict)
        shuffle (bool, optional): If it is necessary to shuffle the dataset. Defaults to True.
        random_state (int, optional):  Defaults to None.

    Returns:
        Train and test split
    """
    skf = StratifiedKFold(n_splits=conf['n_splits'], shuffle=shuffle, random_state=random_state)
    df = update_bboxes(dataset, conf['img_size'])
    df_folds = df[['image_id']].copy()
    
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'object_count'] = df.groupby('image_id')['class_id'].nunique()

    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['object_count'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )

    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

        
    df_folds.reset_index(inplace=True)
    df_valid = pd.merge(df, df_folds[df_folds['fold'] == conf['k_fold']], on='image_id')
    df_train = pd.merge(df, df_folds[df_folds['fold'] != conf['k_fold']], on='image_id')
    
    return df_train, df_valid