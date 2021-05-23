import os
import copy
import argparse
import numpy as np
import pandas as pd
from PIL import Image

def result_ensemble(mask_path, gender_path, age_path, date, save_path):
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    
    mask_model_result = pd.read_csv(mask_path)['ans']
    gender_model_result = pd.read_csv(gender_path)['ans']
    age_model_result = pd.read_csv(age_path)['ans']

    all_predictions = mask_model_result*6 + gender_model_result*3 + age_model_result
    submission['ans'] = all_predictions

    save_path = os.path.join(save_path, 'result')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else: raise
    
    submission.to_csv(os.path.join(save_path, f'{date}_submission.csv'), index=False)
    print('complete!!')

if __name__ == '__main__':
    date = 20210408
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='version_0', help='version name (default: "version_0")')
    args = parser.parse_args()

    save_path = f'/opt/ml/{date}/{args.version}'
    mask_path = os.path.join(save_path, f'{date}_mask_submission.csv')
    gender_path = os.path.join(save_path, f'{date}_gender_submission.csv')
    age_path = os.path.join(save_path, f'{date}_age_submission.csv')
    result_ensemble(mask_path, gender_path, age_path, date, save_path)