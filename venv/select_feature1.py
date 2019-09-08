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

def write_middle_data(data, data_name):
    f = open(data_name, 'w')
    f.write(data)
    f.close()

def read_middle_data(data_name):
    f = open(data_name, 'r')
    return f.read()

def delete_null_feature(ratio):
    # 清除缺失比例高的特征
    delete_cols = []
    null_ratio_dict = json.loads(read_middle_data('null_ratio_dict.txt'))
    cols = null_ratio_dict.keys()
    for col in tqdm(cols):
        # x = sum(df[col].isna().values)
        # null_ratio = x * 1.0 / len(df)
        null_ratio = null_ratio_dict[col]
        # null_ratio_dict[col] = null_ratio
        if null_ratio > ratio:
            delete_cols.append(col)
    return delete_cols
    # write_middle_data(json.dumps(null_ratio_dict), 'null_ratio_dict.txt')

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
    df_train_tran = read_data(r'\train_transaction')
    df_train_iden = read_data(r'\train_identity')
    df_train = df_train_tran.merge(df_train_iden, how = 'left', on = 'TransactionID')
    delete_fea = delete_null_feature(0.5)
    df_train = df_train.drop(delete_fea, axis=1)
    df_train.to_csv(data_path + r'\train_small.csv', index=False)
    # see_feature()

if __name__ == '__main__':
    main()
    # test()



