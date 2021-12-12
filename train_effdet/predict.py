# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import dataset
import model
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm.auto import tqdm
import numpy as np
import pandas as pd 


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            j[0], j[1], j[2][0], j[2][1], j[2][2], j[2][3]))

    return " ".join(pred_strings)

def reshape_boxe(test_df, boxes, img_size, image_id):
    h = test_df[test_df['image_id'] == image_id]['height']
    w = test_df[test_df['image_id'] == image_id]['width']
    for b in boxes :  
        b[0] = (b[0] * w) / img_size 
        b[1] = (b[1] * h) / img_size 
        b[2] = (b[2] * w) / img_size 
        b[3] = (b[3] * h) / img_size 
    return boxes

def predict_effdet(test_df, conf):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    test_ids = test_df.image_id.unique()
    test_dataset = dataset.DatasetRetrieverInference(
                        image_ids=test_ids,
                        marking=test_df,
                        transforms=dataset.get_test_transform()
                        )
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=conf['batch_size'],
            sampler=RandomSampler(test_dataset),
            drop_last=False,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

    net = model.get_effdet_model_inference(conf, use_best=True)

    detection_threshold = conf['detection_threshold']
    img_size = conf['img_size']
    results = []

    with torch.no_grad():

        for images, image_ids in tqdm(test_loader):

            images = torch.stack(images)
            images = images.to(device).float()

            outputs = net(images)

            for i, output in enumerate(outputs):

                image_id = image_ids[i]
                
                boxes = output.detach().cpu().numpy()[:, :4]
                scores = output.detach().cpu().numpy()[:, 4]
                labels = output.detach().cpu().numpy()[:, 5]
                    
                selected = scores >= detection_threshold
                boxes = boxes[selected].astype(np.int32)
                scores = scores[selected]
                labels = labels[selected].astype(np.int32)
                
                if len(boxes)>0 : 
                    labels = [l-1 for l in labels] #re-put the right labels
                    boxes = reshape_boxe(boxes, img_size, image_id)
                    result = {
                        'image_id': image_id,
                        'PredictionString': format_prediction_string(labels, boxes, scores)
                    }
                else :
                    result = {
                        'image_id': image_id,
                        'PredictionString': '14 1.0 0 0 1 1'
                    }


                results.append(result)
                
        torch.cuda.empty_cache()
        del images, image_ids, outputs

    submision_df = pd.DataFrame(results)
    submision_df.to_csv(conf['dir_submision'], index=False)