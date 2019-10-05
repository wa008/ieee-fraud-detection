#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
from tqdm import tqdm
# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from collections import Counter
import xgboost
# myself libs
from kFold_corss import kFold_cross
from write_result import write
from add_other_feature1 import add_other_feature
from read_data import read_all_data
from read_data import read_middle_data
from train import train
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

card_fea = ['card' + str(i) for i in range(1, 7)]
C_fea = ['C' + str(i) for i in range(1, 15)]
D_fea = ['D' + str(i) for i in range(1, 16)]
M_fea = ['M' + str(i) for i in range(1, 10)]
addr_fea = ['addr' + str(i) for i in range(1, 3)]

ext_reduce_fea = []
get_dummies_fea = ['ProductCD', 'P_emaildomain', 'R_emaildomain'] + card_fea + addr_fea + M_fea + ['DeviceType']
ext_plus_fea = ['P_emaildomain'] + M_fea + addr_fea
ext_plus_fea = [x for x in get_dummies_fea]

def delete_null_feature(ratio = 0.4):
    # 清除缺失比例高的特征
    delete_cols = []
    null_ratio_dict = json.loads(read_middle_data(r'\null_ratio_dict.txt'))
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

def delete_both_feature(ratio = 0.4):
    lines = list(read_middle_data(r'\both_single_num_sort.txt').split('\n'))[:-3]
    all_list = []
    for line in lines:
        col = line.split(' ')[0]
        all_list.append(col)
    all_len = len(all_list)
    return all_list[int(all_len * ratio):]

def data_preprocess(df_train_, df_test_, delete_cols, drop_mask):
    df_train = df_train_.copy()
    df_test = df_test_.copy()
    print 'data_preprocess', '-'*100
    print 'df_train, df_test.shape = ', df_train.shape, df_test.shape
    # print df_train.columns.values
    df_train = df_train[~df_train['isFraud'].isna()]
    df_train['train_or_test'] = np.array(['train'] * len(df_train))
    df_test['train_or_test'] = np.array(['test'] * len(df_test))
    df_train_label = df_train[['TransactionID', 'isFraud']]
    df_train = df_train.drop(['isFraud'], axis = 1)
    df = pd.concat([df_train, df_test])
    df = add_other_feature(df)

    print 'in P_emaildomain' in get_dummies_fea
	# 特征增删
    df = df.drop(list(set(delete_cols + ext_reduce_fea) - set(ext_plus_fea)), axis = 1)
    print 'after_delete_plus_reduce_shape = ', df.shape
    print 'df_train, df_test, df shape = ', df_train.shape, df_test.shape, df.shape

    # 缺失值填充
    for col in get_dummies_fea:
        df[col] = df[col].astype(np.str)
    df_train = df[df['train_or_test'] == 'train']
    df_test = df[df['train_or_test'] == 'test']
    cols = df_train.columns.values
    print 'before drop_mask df_train = ', df_train.shape
    if drop_mask == 'fillna_mode':
        for col in tqdm(cols):
            if df_train[col].dtype == 'object':
                df_train[col] = df_train[col].fillna(df_train[col].mode())
            else:
                df_train[col] = df_train[col].fillna(df_train[col].mean())
    elif drop_mask == 'delete_row':
        df_train = df_train.dropna(axis=0)
    print 'after drop_mask df_train = ', df_train.shape
    cols = df_test.columns.values
    for col in tqdm(cols):
        if df_test[col].dtype == 'object':
            df_test[col] = df_test[col].fillna(df_test[col].mode())
        else:
            df_test[col] = df_test[col].fillna(df_test[col].mean())
    df = pd.concat([df_train, df_test])
    print 'in P_emaildomain' in get_dummies_fea

    # one-hot
    print 'one-hot', '-' * 100
    get_dummies_fea_copy = [x for x in get_dummies_fea]
    temp = get_dummies_fea
    for col in temp:
        mid = len(set(df[col].tolist()))
        if mid > 15:
            df = df.drop([col], axis = 1)
            get_dummies_fea_copy.remove(col)
            print 'drop feature = ', col, mid
    print 'before ont-hot df.shape = ', df.shape
    oneHotEncode = OneHotEncoder(sparse=False)
    df_tmp = pd.DataFrame(oneHotEncode.fit_transform(df[get_dummies_fea_copy])).reset_index()
    df = df.reset_index()
    df = df.drop(get_dummies_fea_copy + ['index'], axis = 1)
    df = pd.concat([df, df_tmp], axis = 1)
    cols = df.columns.values
    for col in tqdm(cols):
        df[col] = df[col].astype(np.int, errors = 'ignore')
        df[col] = df[col].astype(np.float, errors = 'ignore')
        if df[col].dtypes == 'object':
            print 'null float col = ', col
    print 'after ont-hot df.shape = ', df.shape

    df_train = df[df['train_or_test'] == 'train']
    df_test = df[df['train_or_test'] == 'test']
    df_train = df_train.merge(df_train_label, how = 'left', on = 'TransactionID')

    df_train = df_train.drop(['train_or_test', 'TransactionID'], axis = 1)
    df_test = df_test.drop(['train_or_test', 'TransactionID'], axis = 1)
    return df_train, df_test

def main():
    time_now = time.time()
    # df_train_glo, df_test_glo = read_all_data('small')
    df_train_glo, df_test_glo = read_all_data('all')
    global get_dummies_fea
    must_delete = delete_null_feature(0.4)

    best_loss = 10000000
    best_ratio = 0.8
    best_mask = 'delete_row'
    delete_cols = delete_both_feature(best_ratio) + must_delete
    df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, best_mask)
    pred = train(df_train, df_test)
    write(pred, '1004_2')
    for ratio in [0.8, 0.5, 0.3, 0.2]:
    # for ratio in []:
        delete_cols = delete_both_feature(ratio) + must_delete
        for drop_mask in ['fillna_mode', 'delete_row']:
            print '\n\ndelete_null_ratio = ', ratio, 'drop_mask = ', drop_mask, '-'*100
            df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, drop_mask)
            for model in [xgboost.XGBRegressor()]:
                loss, mse = kFold_cross(df_train, model) # 0.97
                if best_loss > loss:
                    best_loss = loss
                    best_mask = drop_mask
                    best_ratio = ratio
                print time.time() - time_now, '\n\n'

    delete_cols = delete_both_feature(best_ratio) + must_delete
    df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, best_mask)
    pred = train(df_train, df_test)
    write(pred, '1004_3')

if __name__ == '__main__':
    main()
    # test()

