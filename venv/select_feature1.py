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
    # nrows = 1000
    # df = pd.read_csv(data_path + data_name + '.csv', nrows = nrows)
    df = pd.read_csv(data_path + data_name + '.csv')
    print data_name[1:] + '_shape = ', df.shape, 'read_time = ', time.time() - now
    return df

def write_middle_data(data, data_name):
    f = open(data_name, 'w')
    f.write(data)
    f.close()

def read_middle_data(data_name):
    f = open(data_name, 'r')
    return f.read()

def see_feature():
    now = time.time()
    d = json.loads(read_middle_data('null_ratio_dict.txt'))
    eps = 0.05
    ans = [[] for i in range(int(1.0 / eps))]
    for x, y in d.items():
        ans[int(y / eps)].append(x)
    see_feas = range(10)
    for i in range(10):
        print ''
        print i, i*eps, '-', i*eps + eps, len(ans[i])
        if len(ans[i]) > 0 and i in see_feas:
            print ans[i]



def main():
    see_feature()

if __name__ == '__main__':
    main()
    # test()


