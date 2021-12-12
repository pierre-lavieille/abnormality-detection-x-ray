# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import torch
import model
import dataset
import prepare_data
import time
from tqdm.auto import tqdm
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset,DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

def train_effdet(df, conf):
    """
    train the effdet model and save the best and last weight 
    """
    # define the device to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # define the train and test set and their ids
    train_set, valid_set = prepare_data.k_fold_split(df, conf)
    train_ids = train_set.image_id.unique()
    valid_ids = valid_set.image_id.unique()

    # get the train dataloader
    train_dataset = dataset.DatasetRetriever(
                        image_ids=train_ids,
                        marking=train_set,
                        transforms=dataset.get_train_transforms(),
                        cutmix=True,
                        )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=conf['batch_size'],
            sampler=RandomSampler(train_dataset),
            pin_memory=False,
            drop_last=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

    # get the test dataloader
    valid_dataset = dataset.DatasetRetriever(
                        image_ids=valid_ids,
                        marking=valid_set,
                        transforms=dataset.get_valid_transforms(),
                        cutmix=False,
                        )
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=conf['batch_size'],
            sampler=RandomSampler(valid_dataset),
            pin_memory=False,
            drop_last=True,
            num_workers=4,
            collate_fn=collate_fn,
        )


    # get the model, the optimizer and the lr_scheduler
    effdet = model.get_effdet_model_train(conf)
    optimizer = torch.optim.AdamW(effdet.parameters(), lr=3e-4, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200,
                                T_mult=1,
                                eta_min=1e-6,
                                last_epoch=-1,
                                verbose=False)

    # define a dict to stoch train variables
    history_dict = {}
    history_dict['epoch'] = []
    history_dict['train_loss'] = []
    history_dict['train_lr'] = []
    history_dict['val_loss'] = []
    best_val_summary_loss = 100

    # train the model threw epochs 
    for epoch in range(conf['num_epochs']):
        logger.info('Epoch {}/{}'.format(epoch + 1, conf['num_epochs']))
        logger.info('-' * 10)
        
        history_dict['epoch'].append(epoch)
        
        effdet.train()
        summary_loss = AverageMeter()
        t = time.time()
        loss_trend = []
        lr_trend = []

        # train part
        for images, targets, image_ids in tqdm(train_loader):
            images = torch.stack(images)
            images = images.to(device).float()
            
            target_res = {}
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]
            target_res['bbox'] = boxes
            target_res['cls'] = labels
            optimizer.zero_grad()
            output = effdet(images, target_res)
            
            loss = output['loss']
            loss.backward()
            summary_loss.update(loss.detach().item(), conf['batch_size'])
            optimizer.step()
            
            lr_scheduler.step()
            
            lr = optimizer.param_groups[0]['lr']
            loss_trend.append(summary_loss.avg)
            lr_trend.append(lr)
        history_dict['train_loss'].append(loss_trend)
        history_dict['train_lr'].append(lr_trend)
        logger.info(f'[RESULT]: Train. Epoch: {epoch+1}, summary_loss: {summary_loss.avg:.5f}')

        # validation part
        effdet.eval()
        val_summary_loss = AverageMeter()
        loss_trend = []
        for images, targets, image_ids in tqdm(valid_loader):
            with torch.no_grad():
                images = torch.stack(images)
                images = images.to(device).float()
                target_res = {}
                boxes = [target['boxes'].to(device).float() for target in targets]
                labels = [target['labels'].to(device).float() for target in targets]
                target_res['bbox'] = boxes
                target_res['cls'] = labels 
                target_res["img_scale"] = torch.tensor([1.0] * conf['batch_size'],
                                                        dtype=torch.float).to(device)
                target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * conf['batch_size'],
                                                        dtype=torch.float).to(device)
                    
                output = effdet(images, target_res)
                
                loss = output['loss']
                val_summary_loss.update(loss.detach().item(), conf['batch_size'])

                loss_trend.append(val_summary_loss.avg)
        history_dict['val_loss'].append(loss_trend)

        # save the new weights and best weights
        if val_summary_loss.avg < best_val_summary_loss:
            best_val_summary_loss = val_summary_loss.avg
            effdet.eval()
            torch.save(effdet.state_dict(), conf['dir_best'])
            logger.info('*** NEW BEST ACCURACY ***')
        effdet.eval()
        torch.save(effdet.state_dict(), conf['dir_last'])
        

        logger.info(f'[RESULT]: Valid. Epoch: {epoch+1}, val_summary_loss: {val_summary_loss.avg:.5f}')
    
    return history_dict