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
    df_sub = read_data(r'\sample_submission')
    df_sub['isFraud'] = list(pred)
    df_sub.to_csv(data_path + '\sub_ratio.csv', index = False)
    print 'pred_mean = ', np.mean(pred)
    print 'pred_max = ', np.max(pred)
    print 'pred_min = ', np.min(pred)
    print 'pred_ave = ', np.average(pred)
    print 'pred_median = ', np.median(pred)
    print 'pred_st = ', np.std(pred)

    pred = [1 if x >= 0.5 else 0 for x in pred]
    df_sub['isFraud'] = list(pred)
    df_sub.to_csv(data_path + '\sub_01.csv', index = False)
    print 'number_0 = %d, number_1 = %d, all_data = %d' % (len(pred) - sum(pred), sum(pred), len(pred))
def change_pred():
    df_sub = read_data_csv(r'\sub')
    pre = df_sub['isFraud'].tolist()
    print df_sub.info()
    ans = []
    for x in pre:
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        ans.append(x)
    df_sub['isFraud'] = ans
    df_sub.to_csv(data_path + '\sub_ratio_0_1.csv', index = False)

change_pred()
