# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import timm
import torch

def get_effnet_model_train(conf):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=conf['num_classes'])
    model = model.to(device)
    return model

def get_effnet_model_inference(conf, use_best=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False, num_classes=conf['num_classes'])
    model = model.to(device)
    if use_best :
        checkpoint = torch.load(conf['dir_best'])
    else :
        checkpoint = torch.load(conf['dir_last'])
    model.load_state_dict(checkpoint)
    return model