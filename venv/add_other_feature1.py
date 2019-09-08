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

def read_data(data_name):
    now = time.time()
    nrows = 1000
    df = pd.read_csv(data_path + data_name + '.csv', nrows = nrows)
    # df = pd.read_csv(data_path + data_name + '.csv')
    print data_name[1:] + '_shape = ', df.shape, 'read_time = ', time.time() - now
    return df

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
    return df