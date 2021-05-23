#!/usr/bin/env python
# coding: utf-8

import os
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms

from sklearn.model_selection import train_test_split
'''
Train mode dataloader

'''

class MyDataset(Dataset):
    def __init__(self, images, target_infos, transform):
        self.images = images
        self.target_infos = target_infos
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]

        if self.transform:
            image = self.transform(image)
        return image, self.target_infos[index]

    def __len__(self):
        return len(self.images)

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.target_infos[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def load_outlier(Task):
    mask_outlier = ['000070',
                    '000273',
                    '000645',
                    '003574',
                    '004489',
                    '004490']

    gender_outlier = ['000299',
                      '001266',
                      '001498',
                      '001720',
                      '003421',
                      '003456',
                      '004432',
                      '005223',
                      '006359',
                      '006360',
                      '006361',
                      '006362',
                      '006363',
                      '006364',
                      '006529',
                      '006578']
    age_outlier = []
    
    if Task == 'mask': return mask_outlier
    elif Task == 'gender': return gender_outlier
    elif Task == 'age': return age_outlier
    elif Task == 'all': return list(set(mask_outlier+gender_outlier+age_outlier))
    else:
        raise NameError(f'!!!!! Task ERROR : {Task} !!!!!')
        
def get_gender_ans_category(gender):
    cat = ''
    if gender=='male': cat = 0
    else : cat = 1
    return cat

def get_age_ans_category(age):
    cat = ''
    if int(age) < 30: cat = 0
    elif int(age) < 60: cat = 1
    else : cat = 2
    return cat

def update_index_list(idx):
    new_idx = []
    for idx_list in map(lambda x:list(range(x*7, (x+1)*7)), idx):
        new_idx += idx_list
    return new_idx

def train_valid_index(train_dir, Task, USE_except_outlier, val_ratio, random_seed):
    df = pd.read_csv(os.path.join(train_dir, 'train.csv'))

    if USE_except_outlier:
        df.set_index('id', inplace=True)
        outlier_list = load_outlier(Task)

        df.drop(outlier_list, inplace = True)
        df.reset_index(inplace = True)
    
    df['ans'] = 3*df['gender'].apply(lambda x: get_gender_ans_category(x)) + df['age'].apply(lambda x: get_age_ans_category(x)) 
    
    index_list = list(range(len(df)))
    train_idx, valid_idx, _, _ = train_test_split(index_list, df['ans'], test_size=val_ratio, random_state=random_seed)

    train_idx = update_index_list(train_idx)
    valid_idx = update_index_list(valid_idx)
    return train_idx, valid_idx

def set_train_dataloader(Task, train_argumentation, valid_argumentation, random_seed, val_ratio=0.2, BATCH_SIZE=64, USE_except_outlier=True, USE_imbalanced_sampler=True, USE_DATA_SPLIT=True):
    train_dir = '/opt/ml/input/data/train'
    tf_df = pd.read_csv(os.path.join(train_dir, 'new_info.csv'))
    
    if USE_except_outlier:
        tf_df['id'] = tf_df['path'].apply(lambda x : x.split('/')[7].split('_')[0])
        
        tf_df.set_index('id', inplace=True)
        outlier_list = load_outlier(Task)

        tf_df.drop(outlier_list, inplace = True)
        tf_df.reset_index(inplace = True)
    
    ## load all dataset
    dataset = []
    for path in tf_df['path']:
        image = Image.open(path)
        dataset.append(copy.deepcopy(image))
        image.close()
    
    ## set transform
    train_transform = transforms.Compose(train_argumentation)
    valid_transform = transforms.Compose(valid_argumentation)

    if USE_DATA_SPLIT:
        ## data split
        train_idx, valid_idx = train_valid_index(train_dir, Task, USE_except_outlier, val_ratio=val_ratio, random_seed=random_seed)
        train_datasets, valid_datasets = Subset(dataset, train_idx), Subset(dataset, valid_idx)

        if Task == 'mask':
            train_labels, valid_labels = Subset(tf_df['mask_ans'], train_idx), Subset(tf_df['mask_ans'], valid_idx)
        elif Task == 'gender':
            train_labels, valid_labels = Subset(tf_df['gender_ans'], train_idx), Subset(tf_df['gender_ans'], valid_idx)
        elif Task == 'age':
            train_labels, valid_labels = Subset(tf_df['age_ans'], train_idx), Subset(tf_df['age_ans'], valid_idx)
        #     train_labels, valid_labels = Subset(tf_df['ans'], train_idx), Subset(tf_df['ans'], valid_idx)
        elif Task == 'all':
            train_labels, valid_labels = Subset(tf_df['ans'], train_idx), Subset(tf_df['ans'], valid_idx)
        else:
            raise NameError(f'!!!!! Task ERROR : {Task} !!!!!')
    else:
        train_datasets, valid_datasets = dataset, dataset
        if Task == 'mask':
            train_labels, valid_labels = tf_df['mask_ans'], tf_df['mask_ans']
        elif Task == 'gender':
            train_labels, valid_labels = tf_df['gender_ans'], tf_df['gender_ans']
        elif Task == 'age':
            train_labels, valid_labels = tf_df['age_ans'], tf_df['age_ans']
        #     train_labels, valid_labels = Subset(tf_df['ans'], train_idx), Subset(tf_df['ans'], valid_idx)
        elif Task == 'all':
            train_labels, valid_labels = tf_df['ans'], tf_df['ans']
        else:
            raise NameError(f'!!!!! Task ERROR : {Task} !!!!!')        
    
    ## dataset 생성
    train_datasets = MyDataset(train_datasets, train_labels, train_transform)
    valid_datasets = MyDataset(valid_datasets, valid_labels, valid_transform)

    ## data loader 생성
    if USE_imbalanced_sampler:
        imbalanced_sampler = ImbalancedDatasetSampler(train_datasets)
        train_loader = DataLoader(train_datasets, sampler=imbalanced_sampler, batch_size=BATCH_SIZE, num_workers=4)
    else:
        train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    datasets = {'train':train_datasets, 'valid':valid_datasets}
    dataloaders = {'train':train_loader, 'valid':valid_loader}
    return dataloaders, datasets

