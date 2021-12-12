# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

from effdet import get_efficientdet_config, create_model_from_config
import torch


def get_effdet_model_train(conf):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    base_config = get_efficientdet_config('tf_efficientdet_d4')
    base_config.image_size = (conf['image_size'], conf['image_size'])
    model = create_model_from_config(base_config, bench_task='train', 
                                 bench_labeler=True,
                                 num_classes=conf['num_classes']) 
    model.to(device)
    return model

def get_effdet_model_inference(conf, use_best=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    base_config = get_efficientdet_config('tf_efficientdet_d4')
    base_config.image_size = (conf['image_size'], conf['image_size'])
    net = create_model_from_config(base_config, bench_task='predict', bench_labeler=True,
                                   num_classes=conf['num_classes'])
    net.to(device)

    if use_best :
        checkpoint = torch.load(conf['dir_best'])
    else :
        checkpoint = torch.load(conf['dir_last'])
    net.load_state_dict(checkpoint)
    return net