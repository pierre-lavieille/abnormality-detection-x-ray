# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import prepare_data
import dataset
import model
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
import time
from tqdm.auto import tqdm


def train_effnet(conf):
    # import the train and validation dataframe 
    df_train, df_val = prepare_data.get_update_df(conf)

    # transform them to torch dataframe 
    train_dataset = dataset.Xray(df_train, dataset.get_train_transform(conf), mixup=True)
    val_dataset = dataset.Xray(df_val, dataset.get_valid_transform(), mixup=False)

    # get the loader 
    batch_size = conf['batch_size']

    train_data_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=4,
        batch_size = batch_size
    )

    val_data_loader = DataLoader(
        val_dataset,
        shuffle=True,
        num_workers=4,
        batch_size = batch_size
    )

    # import the model, get the optimizer and the lr scheduler
    effnet = model.get_effnet_model_train(conf)
    optimizer = optim.Adam([param for param in effnet.parameters() if param.requires_grad],
                        lr=conf['lr'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500,
                                T_mult=1,
                                eta_min=1e-6,
                                last_epoch=-1,
                                verbose=False)

    # import some usefull torch fonction
    criterion = nn.BCEWithLogitsLoss()#.to(device)
    sigmoid = nn.Sigmoid()

    # initialize some tools
    since = time.time()
    best_acc = 0.0
    history_dict = {}
    history_dict['epoch'] = []
    history_dict['train_loss'] = []
    history_dict['train_acc'] = []
    history_dict['train_lr'] = []
    history_dict['val_loss'] = []
    history_dict['val_acc'] = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    logger.info(" Begining of the training Process")
    for epoch in range(conf['num_epochs']):
        logger.info('Epoch {}/{}'.format(epoch + 1, conf['num_epochs']))
        logger.info('-' * 10)

        history_dict['epoch'].append(epoch)
        
        effnet.train()  # Set model to training mode

        current_loss = 0.0
        current_corrects = 0

        # Here's where the training happens
        logger.info('Iterating through data...')

        for images, targets in tqdm(train_data_loader):
            images = images.to(device)
            targets = targets.reshape(batch_size, 1).to(device)
            outputs = effnet(images)
            targets = targets.type_as(outputs)

            # We need to zero the gradients, don't forget it
            optimizer.zero_grad()

            # Time to carry out the forward training poss
            # We only need to log the loss stats if we are in training phas
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # We want variables to hold the loss statistics
            current_loss += loss.item() * images.size(0)
            preds = torch.round(sigmoid(outputs))
            current_corrects += torch.sum(preds == targets.data)
            
            lr_scheduler.step()
            history_dict['train_lr'].append(optimizer.param_groups[0]['lr'])

        epoch_loss = current_loss / len(train_data_loader.dataset)
        epoch_acc = current_corrects / len(train_data_loader.dataset)  
        history_dict['train_acc'].append(epoch_acc)
        history_dict['train_loss'].append(epoch_loss)
        logger.info('TRAIN Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
        
        effnet.eval()  # Set model to validation mode
        current_loss_val = 0.0
        current_corrects_val = 0
        
        # free some space
        del images, targets, outputs
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for images2, targets2 in tqdm(val_data_loader):
                images2 = images2.to(device)
                targets2 = targets2.reshape(batch_size, 1).to(device)
                outputs2 = effnet(images2)
                targets2 = targets2.type_as(outputs2)

                #calculate the loss
                loss2 = criterion(outputs2, targets2)

                # Get the accuracy
                current_loss_val += loss2.item() * images2.size(0)
                preds2 = torch.round(sigmoid(outputs2))
                current_corrects_val += torch.sum(preds2 == targets2.data)
        
        # free some space
        del images2, targets2, outputs2
        torch.cuda.empty_cache()
        
        epoch_loss_val = current_loss_val / len(val_data_loader.dataset)
        epoch_acc_val = current_corrects_val / len(val_data_loader.dataset)
        history_dict['val_acc'].append(epoch_acc_val)
        history_dict['val_loss'].append(epoch_loss_val)
        logger.info('VALIDATION Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss_val, epoch_acc_val))
        
        # same the last of potential best weight
        effnet.eval()
        torch.save(effnet.state_dict(), conf['dir_last'])
        if epoch_acc_val>best_acc :
            logger.info('*** NEW BEST ACCURACY ***')
            best_acc = epoch_acc_val
            torch.save(effnet.state_dict(), conf['dir_best'])

    time_since = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_since // 60, time_since % 60))

    return history_dict
