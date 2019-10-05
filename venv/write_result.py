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
from read_data import read_data_csv
data_path = r'D:\kaggle\data\ieee-fraud-detection'


def write(pred, time_name):
    print 'write', '-' * 100
    df_sub = read_data_csv(r'\sample_submission')
    df_sub['isFraud'] = list(pred)
    df_sub.to_csv(data_path + '\sub_ratio_' + time_name + '.csv', index = False)
    print 'pred_mean = ', np.mean(pred)
    print 'pred_max = ', np.max(pred)
    print 'pred_min = ', np.min(pred)
    print 'pred_ave = ', np.average(pred)
    print 'pred_median = ', np.median(pred)
    print 'pred_st = ', np.std(pred)

# change_pred()
