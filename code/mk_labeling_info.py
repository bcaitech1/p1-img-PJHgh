import os
import numpy as np
import pandas as pd
from PIL import Image


def get_age_category(age):
    cat = ''
    if int(age) < 30: cat = '< 30'
    elif int(age) < 60: cat = '>= 30 and < 60'
    else : cat = '>= 60'
    return cat

def get_mask_category(mask):
    cat = ''
    mask = mask.split('.')[0]
    if mask == 'incorrect_mask': cat = 'incorrect'
    elif mask == 'normal': cat = 'normal'
    else : cat = 'wear'
    return cat

def get_ans_category(mask, gender, age):
    cat = 0
    mask = mask.split('.')[0]
    if mask == 'incorrect_mask': cat += 6
    elif mask == 'normal': cat += 12
    
    if gender == 'female': cat += 3
    
    if int(age) >= 30 and int(age) < 60: cat += 1
    elif int(age) >= 60: cat += 2
    return cat
    
def get_mask_ans_category(mask):
    cat = ''
    if mask=='wear': cat = 0
    elif mask=='incorrect': cat = 1
    else : cat = 2
    return cat

def get_gender_ans_category(gender):
    cat = ''
    if gender=='male': cat = 0
    else : cat = 1
    return cat

def get_age_ans_category(age):
    cat = ''
    if age=='< 30': cat = 0
    elif age=='>= 30 and < 60': cat = 1
    else : cat = 2
    return cat

if __name__ == '__main__':
    train_dir = '/opt/ml/input/data/train'
    train_csv = pd.read_csv(os.path.join(train_dir, 'train.csv'))

        
    tf_df = pd.DataFrame({'mask':[], 'gender':[], 'age':[], 'path':[], 'ans':[]})
    tr_image = os.path.join(train_dir, 'images')
    idx = 0
    for id_path in train_csv['path']:
        _, gender, _, age = list(id_path.split('_'))
        id_path = os.path.join(tr_image, id_path)
        for mask in os.listdir(id_path):
            if mask[0] == '.': continue
            path = os.path.join(id_path, mask)
            ans = get_ans_category(mask, gender, age)
            tf_df.loc[idx]=[get_mask_category(mask), gender, get_age_category(age), path, ans]
            idx += 1
    
    tf_df['mask_ans'] = tf_df['mask'].apply(lambda x : get_mask_ans_category(x))
    tf_df['gender_ans'] = tf_df['gender'].apply(lambda x : get_gender_ans_category(x))
    tf_df['age_ans'] = tf_df['age'].apply(lambda x : get_age_ans_category(x))

    tf_df.to_csv(os.path.join(train_dir, 'new_info.csv'), index=False)