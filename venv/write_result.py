#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
# machine learning
import xgboost
# myself libs
from kFold_corss import kFold_cross

# other
import warnings
warnings.filterwarnings("ignore")
from change_format import read_data_csv
data_path = r'D:\kaggle\data\ieee-fraud-detection'


def write(pred):
    df_sub = read_data_csv(r'\sample_submission')
    df_sub['isFraud'] = list(pred)
    df_sub.to_csv(data_path + '\sub_ratio.csv', index = False)
    print 'pred_mean = ', np.mean(pred)
    print 'pred_max = ', np.max(pred)
    print 'pred_min = ', np.min(pred)
    print 'pred_ave = ', np.average(pred)
    print 'pred_median = ', np.median(pred)
    print 'pred_st = ', np.std(pred)
    pred = np.maximum(pred, 0)
    pred = np.minimum(pred, 1)
    df_sub['isFraud'] = list(pred)
    df_sub.to_csv(data_path + '\sub_01.csv', index = False)
    print 'ypred_average =',np.average(list(pred))

# change_pred()
