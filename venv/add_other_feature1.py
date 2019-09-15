#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
from tqdm import tqdm
# machine learning
import xgboost
# myself libs
from kFold_corss import kFold_cross
from write_result import write
# other
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'


def add_other_feature(df):
    amt_list = df['TransactionAmt'].tolist()
    amt_float_list = []
    for x in amt_list:
        s = str(x).split('.')
        if len(s) == 2:
            # if s[1][-1] == '0':
            #     print 's[1] = ', s[1]
            amt_float_list.append(len(s[1]))
        else:
            assert len(s) == 1
            amt_float_list.append(0)
    df['TransactionAmt_float_num'] = amt_float_list
    dt_list = df['TransactionDT'].tolist()
    wed_list = []
    for x in dt_list:
        try:
            wed_list.append(int(x) / 60 / 60 / 24)
        except:
            wed_list.append(np.nan)
    df['week'] = wed_list
    return df

