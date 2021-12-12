# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import pandas as pd 
import dataset
import model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def predict_effnet(conf):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # import the test dataframe where the id of the test images are 
    test_df = pd.read_csv(conf['dir_df_test'])

    # create the trch dataset
    test_dataset = dataset.Xray_val(test_df, conf['DIR_TEST'], transform=dataset.get_valid_transform())

    # create the dataloader
    test_data_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=4,
        batch_size = 1
    )

    # import the model with the best weights
    net = model.get_effnet_model_inference(conf, use_best=True)
    net.eval()
    net.to(device)

    # initialize some tools
    results = []
    sigmoid = nn.Sigmoid()

    # sub-optimal inference process
    # we only use one workers this way 
    logger.info('Strat of the Inference')
    with torch.no_grad():
        for images, image_ids in tqdm(test_data_loader):
            
            images = images.to(device)
            outputs = model(images)
            preds = sigmoid(outputs).data.cpu().numpy()
            preds = round(preds[0][0], 6)
            
            result = {'image_id': image_ids[0],
                    'Prediction': preds}
            results.append(result)
    logger.info('Enf of the Inference')

    submision = pd.DataFrame(results)
    submision.to_csv(conf['dir_submision'], index=False)
