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
import xgboost
# myself libs
from kFold_corss import kFold_cross
from write_result import write
from add_other_feature1 import add_other_feature
# other
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'
pd.set_option('display.max_columns', None)

def read_data(data_name, mark):
    now = time.time()
    df = ''
    if mark == 'small':
        nrows = 5000
        df = pd.read_csv(data_path + data_name + '.csv', nrows = nrows)
    elif mark == 'all':
        df = pd.read_csv(data_path + data_name + '.csv')
    # print data_name[1:] + '_shape = ', df.shape, 'read_time = ', time.time() - now
    # print 'df.info() = ', df.info()
    return df

def write_middle_data(data, data_name):
    f = open(data_name, 'w')
    f.write(data)
    f.close()

def read_middle_data(data_name):
    f = open(data_name, 'r')
    return f.read()

card_fea = ['card' + str(i) for i in range(1, 7)]
C_fea = ['C' + str(i) for i in range(1, 15)]
D_fea = ['D' + str(i) for i in range(1, 16)]
M_fea = ['M' + str(i) for i in range(1, 10)]
addr_fea = ['addr' + str(i) for i in range(1, 3)]

ext_reduce_fea = []
get_dummies_fea = ['ProductCD', 'P_emaildomain', 'R_emaildomain'] + card_fea + addr_fea + M_fea + ['DeviceType']
ext_plus_fea = ['P_emaildomain'] + M_fea + addr_fea
ext_plus_fea = get_dummies_fea

def read_all_data(mark = 'small'):
    print 'read_all_data', '-'*100
    now = time.time()
    df_train_tran = read_data(r'\train_transaction', mark)
    df_train_iden = read_data(r'\train_identity', mark)
    df_test_tran = read_data(r'\test_transaction', mark)
    df_test_iden = read_data(r'\test_identity', mark)
    df_train = df_train_tran.merge(df_train_iden, how = 'left', on = 'TransactionID')
    df_test = df_test_tran.merge(df_test_iden, how = 'left', on = 'TransactionID')
    print 'read_data_time = ', time.time() - now
    return df_train, df_test

def delete_null_feature(ratio = 0.4):
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

def delete_both_feature(ratio = 0.4):
    lines = list(read_middle_data('both_single_num_sort.txt').split('\n'))[:-3]
    all_list = []
    for line in lines:
        col = line.split(' ')[0]
        num = int(line.split(' ')[1])
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
        
    # one-hot
    print 'one-hot', '-' * 100
    for col in get_dummies_fea:
        print col + "_setLen = ", len(set(df[col].tolist()))
    # print 'df.columns.values = ', sorted(df.columns.values)
    # print 'get_dummies_fea = ', get_dummies_fea
    # print 'error_feature = ', set(get_dummies_fea) - set(df.columns.values)
    print 'before ont-hot df.shape = ', df.shape
    for i in tqdm(range(len(get_dummies_fea))):
        # print 'i = ', i, get_dummies_fea[i]
        if get_dummies_fea[i] in ['card1',]: continue
        df = pd.get_dummies(df, columns=[get_dummies_fea[i]])
    for col in get_dummies_fea:
        if col in df.columns.values:
            print 'drop col = ', col
            df = df.drop([col], axis = 1)
    cols = df_test.columns.values
    for col in tqdm(cols):
        if df_test[col].dtype != 'object': 
            continue
        try:
            df_test[col] = df_test[col].astype(np.float)
        except:
            print 'null float col = ', col
    print 'after ont-hot df.shape = ', df.shape

    df_train = df[df['train_or_test'] == 'train']
    df_test = df[df['train_or_test'] == 'test']
    df_train = df_train.merge(df_train_label, how = 'left', on = 'TransactionID')

    df_train = df_train.drop(['train_or_test', 'TransactionID'], axis = 1)
    df_test = df_test.drop(['train_or_test', 'TransactionID'], axis = 1)
    # print "df_train.info() = ", df_train.info()
    # print "df_test.info() = ", df_test.info()
    return df_train, df_test


def train(df_train, df_test):
    # kFold_cross(df_train, 5, 'isFraud')
    label = np.array(df_train['isFraud'].tolist())
    train = df_train.drop(['isFraud'], axis = 1).values
    test = df_test.values
    xgb = xgboost.XGBRegressor()
    # xgb = RandomForestRegressor()
    print 'train.shape, test.shape, label.shape = ', train.shape, test.shape, label.shape
    xgb.fit(train, label)
    pred = xgb.predict(test)
    return pred

def test():
    df_train_tran = read_data(r'\train_transaction')
    train_cols = set(df_train_tran.columns.values)
    df_test_tran = read_data(r'\test_transaction')
    test_cols = set(df_test_tran.columns.values)
    print train_cols - test_cols

def main():
    time_now = time.time()
    # df_train_glo, df_test_glo = read_all_data('small')
    df_train_glo, df_test_glo = read_all_data('all')

    must_delete = delete_null_feature(0.4)

    best_loss = 10000000
    best_ratio = 0.4
    best_mask = 'delete_row'
    for ratio in [0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
    # for ratio in []:
        delete_cols = delete_both_feature(ratio) + must_delete
        for drop_mask in ['fillna_mode', 'delete_row']:
            print '\n\ndelete_null_ratio = ', ratio, 'drop_mask = ', drop_mask, '-'*100
            df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, drop_mask)
            for model in [xgboost.XGBRegressor()]:
                loss = kFold_cross(df_train, model) # 0.97
                if best_loss > loss:
                    best_loss = loss
                    best_mask = drop_mask
                    best_ratio = ratio
                print time.time() - time_now, '\n\n'

    delete_cols = delete_both_feature(best_ratio) + must_delete
    df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, best_mask)
    pred = train(df_train, df_test)
    write(pred, '0915_1')

if __name__ == '__main__':
    main()
    # test()

