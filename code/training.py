#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import copy
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision import models
try:
    from efficientnet_pytorch import EfficientNet
except:
    RuntimeError('you need to install efficientnet_pytorch library (method : "pip install efficientnet_pytorch")')
try:
    from madgrad import MADGRAD
except:
    RuntimeError('you need to install madgrad library (method : "pip install madgrad")')

from loss import *
from inference import *
from dataloader import *
from argumentation import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def set_num_classes(Task):
    if Task == 'mask':
        num_classes = 3
    elif Task == 'gender':
        num_classes = 2
    elif Task == 'age':
        num_classes = 3
    elif Task == 'all':
        num_classes = 18
    else:
        raise NameError(f'!!!!! Task ERROR : {Task} !!!!!')
    return num_classes

def set_model(model_name, num_classes):
    if model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=2048, out_features=num_classes))
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=1024, out_features=num_classes)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
        model.classifier = torch.nn.Linear(in_features=2208, out_features=num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        model.fc = torch.nn.Linear(in_features=1024, out_features=num_classes)
    elif model_name == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained=True)
        model.fc = torch.nn.Linear(in_features=1024, out_features=num_classes)
    elif model_name == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = torch.nn.Linear(in_features=1024, out_features=num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    elif model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
    elif model_name == 'mnasnet0_5':
        model = models.mnasnet0_5(pretrained=True)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    elif model_name == 'mnasnet1_0':
        model = models.mnasnet1_0(pretrained=True)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)    
    elif model_name == 'efficientnet-b0':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name == 'efficientnet-b1':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name == 'efficientnet-b2':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name == 'efficientnet-b4':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name == 'efficientnet-b5':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name == 'efficientnet-b6':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    elif model_name == 'efficientnet-b7':
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else:
        raise NameError(f'!!!!! Model ERROR : {model_name} !!!!!')
    return model

def set_loss(loss, param=None):
    if loss == 'label smoothing':
        criterion = LabelSmoothingLoss(smoothing=0.2)
    elif loss == 'new label smoothing':
        criterion = ModifiedLabelSmoothingLoss(param['device'])
    elif loss == 'cross entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss == 'FocalLoss':
        criterion = FocalLoss()
    else:
        raise NameError(f'!!!!! loss ERROR : {loss} !!!!!')
    return criterion

def set_optimizer(optimizer, model, learning_rate):
    if optimizer == 'Adam':
        optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer_ft = optim.SGD(model.parameters(), 
                                 lr = learning_rate,
                                 momentum=0.9,
                                 weight_decay=1e-4)
    elif optimizer == 'MADGRAD':
        optimizer_ft = MADGRAD(model.parameters(),
                               lr=learning_rate)
    else:
        raise NameError(f'!!!!! operator ERROR : {operator} !!!!!')
    return optimizer_ft

def set_scheduler(scheduler, optimizer_ft):
    if scheduler == 'MultiplicativeLR':
        exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lambda epoch: 0.98739)
    elif scheduler == 'StepLR':
        lr_decay_step = 3
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.5)
    elif scheduler == 'ReduceLROnPlateau':
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor=0.1, patience=10)
    elif scheduler == 'CosineAnnealingLR':
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0.)
    else:
        raise NameError(f'!!!!! scheduler ERROR : {scheduler} !!!!!')
    return exp_lr_scheduler

def train_model(date, Task, model, criterion, optimizer, scheduler, dataloaders, device, num_epochs, f1_factor, save_path, scheduler_name, accumulation_steps, patience):
    since = time.time()
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_f1 = 0.0, 0.0
    train_loss, train_acc, train_f1, train_precision, train_recall = [], [], [], [], []
    valid_loss, valid_acc, valid_f1, valid_precision, valid_recall = [], [], [], [], []
    
    # -- logging
    logger = SummaryWriter(log_dir=save_path)
    with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, running_f1_score, num_cnt = 0.0, 0, 0.0, 0
            running_recall, running_precision = 0.0, 0.0
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.to(device))
                    _, preds = torch.max(outputs, 1)
                    
                    f1_score, recall, precision = f1_loss(labels.to(device), outputs, Task)
                    loss = (1-f1_factor)*criterion(outputs, labels.long().to(device)) + f1_factor*(1 - f1_score)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if (idx+1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    if phase == 'train':
                        logger.add_scalar("Train/loss", loss.item(), epoch*len(dataloaders[phase]) + idx)
                        logger.add_scalar("Train/accuracy", torch.sum(preds.cpu() == labels).item()/inputs.size(0)*100, epoch*len(dataloaders[phase]) + idx)
                        logger.add_scalar("Train/f1_score", f1_score.cpu(), epoch*len(dataloaders[phase]) + idx)
                        logger.add_scalar("Train/precision", precision.cpu(), epoch*len(dataloaders[phase]) + idx)
                        logger.add_scalar("Train/recall", recall.cpu(), epoch*len(dataloaders[phase]) + idx)
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.cpu() == labels)
                running_f1_score += f1_score.cpu()
                running_recall += recall.cpu()
                running_precision += precision.cpu()
                num_cnt += len(labels)
            
            if scheduler_name == 'ReduceLROnPlateau' and phase == 'valid':
                scheduler.step(running_loss / num_cnt)
            elif scheduler_name != 'ReduceLROnPlateau' and phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            epoch_f1 = float(running_f1_score / len(dataloaders[phase]))
            epoch_recall = float(running_recall / len(dataloaders[phase]))
            epoch_precision = float(running_precision / len(dataloaders[phase]))
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                train_f1.append(epoch_f1)
                train_precision.append(epoch_precision)
                train_recall.append(epoch_recall)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                valid_f1.append(epoch_f1)
                valid_precision.append(epoch_precision)
                valid_recall.append(epoch_recall)
                
                logger.add_scalar("Val/loss", epoch_loss, epoch)
                logger.add_scalar("Val/accuracy", epoch_acc, epoch)
                logger.add_scalar("Val/f1_score", epoch_f1, epoch)
                logger.add_scalar("Val/precision", epoch_precision, epoch)
                logger.add_scalar("Val/recall", epoch_recall, epoch)
                
            print(f'\t{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f} F1 score: {epoch_f1:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')
            # deep copy the model
            if phase == 'valid' and epoch_f1 > best_f1:
                best_idx = epoch
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
#                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print(f'\t==> best model saved - {best_idx} / {best_acc:.4f} / {best_f1:.4f}')
#                 torch.save(model.state_dict(), os.path.join(save_path, f'{date}_{Task}_{best_idx}th_epoch_model.pt'))
                counter = 0
            else:
                counter += 1
        if counter > patience:
            print("Early Stopping...")
            break
        
        end_time = time.time() - start_time
        print(f'\t1 EPOCH Training complete in {end_time//60}m {end_time%60:.2f}s', end='\n\n')
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60}m {time_elapsed%60:.2f}s')
    print(f'Best valid Acc: {best_idx} - {best_acc:.4f}')
    print(f'Best valid F1: {best_idx} - {best_f1:.4f}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_path, f'{date}_{Task}_best_model.pt'))
    print('model saved')
    
    train_result = pd.DataFrame({'train_loss':train_loss,
                                 'train_acc':train_acc,
                                 'train_f1':train_f1,
                                 'train_precision':train_precision,
                                 'train_recall':train_recall,
                                 'valid_loss':valid_loss,
                                 'valid_acc':valid_acc,
                                 'valid_f1':valid_f1, 
                                 'valid_precision':valid_precision,
                                 'valid_recall':valid_recall})
    train_result.to_csv(os.path.join(save_path, 'train_result.csv'), index=False)
    return model, device
    
def main(date, save_path, args):
    if args.seed != None:
        seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set gpu
    
    print('#'*100)
    print("{0:#^100}".format(f" {args.task} {args.name} "))
    print('#'*100)
    
    print('-' * 100)
    print('Data loading start !!')
    dataloaders, datasets = set_train_dataloader(Task = args.task,
                                                 train_argumentation = set_argumentation(args.train_augmentation),
                                                 valid_argumentation = set_argumentation(args.valid_augmentation),
                                                 random_seed = args.seed,
                                                 val_ratio = args.val_ratio,
                                                 BATCH_SIZE=args.batch_size,
                                                 USE_except_outlier=args.USE_except_outlier,
                                                 USE_imbalanced_sampler=args.USE_imbalanced_sampler,
                                                 USE_DATA_SPLIT=args.USE_DATA_SPLIT)
    print('\t>>> Data loading complete !!')
    
    print('Train Model settting start !!')
    num_classes = set_num_classes(Task = args.task)
    model = set_model(model_name=args.model,
                      num_classes=num_classes).to(device)
    criterion = set_loss(loss=args.criterion,
                         param={'device':device, 'datasets':datasets})
    optimizer_ft = set_optimizer(optimizer=args.optimizer,
                                 model=model,
                                 learning_rate=args.lr)
    scheduler = set_scheduler(scheduler=args.scheduler, optimizer_ft=optimizer_ft)
    print('\t>>> Train Model settting complete !!')
    
    print('Training start !!')
    print('-' * 100)
    model, device = train_model(date=date,
                                Task=args.task,
                                model=model,
                                criterion=criterion,
                                optimizer=optimizer_ft,
                                scheduler=scheduler,
                                dataloaders=dataloaders,
                                device=device,
                                num_epochs=args.epochs,
                                f1_factor=args.f1_factor,
                                save_path=save_path,
                                scheduler_name=args.scheduler,
                                accumulation_steps=args.accumulation_steps,
                                patience=args.patience)
    print('\t>>> Training complete !!')
    print('-' * 100)
    
    print('inference start !!')
    mk_test_result(model=model,
                   date=date,
                   task=args.task,
                   test_argumentation=set_argumentation(args.valid_augmentation),
                   save_path=save_path,
                   device=device)
    print('\t>>> inference complete !!')
    
    
if __name__ == '__main__':
    date = 20210408
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--task', type=str, default='all', help='task of classification : "mask" or "gender" or "age" or "all"  (default: "all")')
    parser.add_argument('--train_augmentation', type=str, default='version2', help='train data augmentation type (default: version2)')
    parser.add_argument('--valid_augmentation', type=str, default='version4', help='valid data augmentation type (default: version4)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--USE_except_outlier', type=bool, default=True, help='USE except outlier (default: True)')
    parser.add_argument('--USE_imbalanced_sampler', type=bool, default=True, help='USE imbalanced sampler (default: True)')
    parser.add_argument('--USE_DATA_SPLIT', type=bool, default=True, help='USE train, validation data split (default: True)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='data split ratio for validaton (default: 0.2)')
    
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient Accumulation Steps (default: 8)')
    parser.add_argument('--patience', type=int, default=4, help='Early Stopping patience step number (default: 4)')
    parser.add_argument('--f1_factor', type=float, default=0.85, help='f1 loss factor to train (default: 0.5)')
    parser.add_argument('--model', type=str, default='resnext50_32x4d', help='model name (default: resnext50_32x4d)')
    parser.add_argument('--optimizer', type=str, default='MADGRAD', help='optimizer type (default: MADGRAD)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='cross entropy', help='criterion type (default: cross entropy)')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='scheduler type (default: MultiplicativeLR)')
    parser.add_argument('--name', type=str, default='version_0', help='name')
    
    args = parser.parse_args()
    
    save_path = f'/opt/ml/{date}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    save_path = os.path.join(save_path, f'{args.name}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    main(date, save_path, args)
