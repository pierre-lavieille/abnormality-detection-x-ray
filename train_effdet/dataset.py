# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset,DataLoader
import torch
import cv2


# train set data augmentation
def get_train_transforms():
    return A.Compose(
        [
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=25, 
                            border_mode=0, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                            val_shift_limit=0.2, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                            contrast_limit=0.2, p=1.0),
        ],p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0)
            ],p=0.3),
        A.Cutout(num_holes=8, max_h_size=54, max_w_size=54, fill_value=0, p=0.2),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255, p=1.0),
        ToTensorV2(p=1.0)
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

# validation set data augmentation
def get_valid_transforms():
    return A.Compose(
        [
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

# inference data augmentation
def get_test_transform():
    return A.Compose([
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255, p=1.0),
        ToTensorV2(p=1.0)
    ])

class DatasetRetriever(Dataset):

    def __init__(self, marking, TRAIN_ROOT_PATH, image_ids, transforms=None, mixup=False, cutmix=False, img_size=1024):
        super().__init__()
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.cutmix = cutmix
        self.mixup = mixup
        self.TRAIN_ROOT_PATH = TRAIN_ROOT_PATH
        self.img_size = img_size
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        if self.mixup and random.random() > 0.8:
            image, boxes, labels = self.load_mixup_image_and_boxes(index)
        elif self.cutmix and random.random() > 0.8:
            image, boxes, labels = self.load_cutmix_image_and_boxes(index)
        else:
            image, boxes, labels = self.load_image_and_boxes(index)
        image = image.astype(np.uint8)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])
        
        if self.transforms:
            for i in range(10):
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                if len(sample['bboxes'])>0 :
                    break
                
            image = sample['image']
                
            target['boxes'] = torch.tensor(sample['bboxes'])
            #target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            if target['boxes'].shape[0] > 0 :
                target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  ## ymin, xmin, ymax, xmax
            target['labels'] = torch.tensor(sample['labels'])
            target['labels'] += 1 #because zero is count as the background for effdet

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    # classic loading of image and box
    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        img = cv2.imread(f'{self.TRAIN_ROOT_PATH}/{image_id}.png', cv2.IMREAD_COLOR).copy()
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.float32)
        
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = records['class_id'].tolist()
        return image, boxes, labels
   
   # loading mixup images and boxes
    def load_mixup_image_and_boxes(self, index):
        image, boxes, labels = self.load_image_and_boxes(index)
        r_image, r_boxes, r_labels = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        mixup_image = (image+r_image)/2
        return mixup_image, np.vstack((boxes, r_boxes)).astype(np.int32), np.concatenate((labels, r_labels))

    # loading cutmix images and boxes
    def load_cutmix_image_and_boxes(self, index):   
        imsize = self.img_size
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)
            del image, boxes, labels

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])

        result_boxes = result_boxes.astype(np.int32)
        keep = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)
        result_boxes = result_boxes[keep]
        result_labels = result_labels[keep].astype(np.int32)
            
        return result_image, result_boxes, result_labels 


class DatasetRetrieverInference(Dataset):

    def __init__(self, marking, image_ids, TEST_ROOT_PATH,transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.TEST_ROOT_PATH = TEST_ROOT_PATH
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        image_id = self.image_ids[index]
        img = cv2.imread(f'{self.TEST_ROOT_PATH}/{image_id}.png', cv2.IMREAD_COLOR).copy()
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.uint8)
        
        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']    

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]