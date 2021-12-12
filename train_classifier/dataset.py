# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train Dataset 
class Xray(Dataset):
    def __init__(self, df, conf, transform=None, mixup=False):
        self.df = df
        self.transform = transform
        self.mixup = mixup
        self.conf = conf

    def __getitem__(self,idx):
        img_src = self.df.loc[idx,'path']
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image - image.min()
        image = image / image.max()
        
        target = self.df.loc[idx,'target']
        
        if self.mixup and random.random() > self.conf['prob_mixup']: 
            r_idx = random.randint(0, len(self.df) - 1)
            r_img_src = self.df.loc[r_idx,'path']
            r_image = cv2.imread(r_img_src)
            r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB).astype(np.float32)
            r_image = r_image - r_image.min()
            r_image = r_image / r_image.max()
            
            r_target = self.df.loc[r_idx,'target']
            
            image = (image+r_image)/2
            target = (target + r_target)/2 
            
        image = (image * 255.0).astype(np.uint8)  
        target = torch.tensor(target, dtype=torch.float32)
        
        if (self.transform):
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, target 
    
    def __len__(self):
        return(len(self.df))


# Albumentations
def get_train_transform(conf):
    return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(p=0.25, alpha=0.5, sigma=0.25),
                A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=10, p=0.5),
                A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                          contrast_limit=0.2, p=0.5),
                    ],p=0.5),
                A.Cutout(num_holes=8, max_h_size=54, max_w_size=54, fill_value=0, p=0.2),
                A.RandomSizedCrop(min_max_height = [480, 520], height=conf['img_size'], width=conf['img_size'], p=0.2),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ])

def get_valid_transform():
    return A.Compose([
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ])

# Inference Dataset
class Xray_val(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.transform = transform
        self.image_dir = image_dir

    def __getitem__(self,idx):
        image_id = self.df['image_id'][idx]
        image = cv2.imread(f"{self.image_dir}/{image_id}.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image - image.min()
        image = image / image.max()
        image = image * 255.0
        
        if (self.transform):
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, image_id
    
    def __len__(self):
        return(len(self.df))