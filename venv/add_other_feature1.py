#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
from tqdm import tqdm
# from kFold_corss import kFold_cross
# from write_result import write
# other
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'

def point_num(x):
    if '.' in str(x):
        return len(str(x).split('.')[1])
    return 0

def add_other_feature(df):
    df['TransactionAmt_float_num'] = df['TransactionAmt'].apply(point_num)
    df['week'] = df['TransactionDT'].astype(np.int) // (60 * 60 * 24)
    return df

