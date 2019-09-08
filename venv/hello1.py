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
from add_other_feature1 import add_other_feature
# other
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'

def read_data(data_name, mark):
    now = time.time()
    df = ''
    if mark == 'small':
        nrows = 100
        df = pd.read_csv(data_path + data_name + '.csv', nrows = nrows)
    elif mark == 'all':
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

card_fea = ['card' + str(i) for i in range(1, 7)]
C_fea = ['C' + str(i) for i in range(1, 15)]
D_fea = ['D' + str(i) for i in range(1, 16)]
M_fea = ['M' + str(i) for i in range(1, 10)]
addr_fea = ['addr' + str(i) for i in range(1, 3)]
ext_plus_fea = ['P_emaildomain'] + M_fea + addr_fea
ext_reduce_fea = []
get_dummies_fea = ['ProductCD', 'P_emaildomain'] + card_fea + addr_fea + M_fea

def read_all_data(mark):
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

    df = df.drop(list(set(delete_cols + ext_reduce_fea) - set(ext_plus_fea)), axis  =1)
    print 'after_delete_plus_reduct_shape = ', df.shape
    print 'df_train, df_test, df shape = ', df_train.shape, df_test.shape, df.shape
    # 特征类型转换
    df_train = df[df['train_or_test'] == 'train']
    df_test = df[df['train_or_test'] == 'test']
    cols = df_train.columns.values
    print 'before drop_mask df_train = ', df_train.shape
    if drop_mask == 'fillna_mode':
        for col in tqdm(cols):
            df_train[col] = df_train[col].fillna(df_train[col].mode())
    elif drop_mask == 'delete_row':
        df_train = df_train.dropna(axis=0)
    print 'after drop_mask df_train = ', df_train.shape
    cols = df_test.columns.values
    for col in tqdm(cols):
        df_test[col] = df_test[col].fillna(df_test[col].mode())
    df = pd.concat([df_train, df_test])
    print sorted(df.columns.values)
    print get_dummies_fea
    print set(get_dummies_fea) - set(df.columns.values)
    print 'before ont-hot = ', df.shape
    for col in get_dummies_fea:
        df[col] = df[col].astype(np.str)
    for i in tqdm(range(len(get_dummies_fea))):
        print 'i = ', i, get_dummies_fea[i]
        if get_dummies_fea[i] in ['card1',]: continue
        df = pd.get_dummies(df, columns=[get_dummies_fea[i]])
    # df = pd.get_dummies(df, columns=get_dummies_fea)
    for col in get_dummies_fea:
        if col not in ['card1', ] and col in df.columns.values:
            print 'drop col = ', col
            df = df.drop([col], axis = 1)
    print df.info()
    print 'after ont-hot = ', df.shape
    cols = df_test.columns.values
    for col in tqdm(cols):
        try:
            df_test[col] = df_test[col].astype(np.float)
        except:
            print 'null float col = ', col

    df_train = df[df['train_or_test'] == 'train']
    df_test = df[df['train_or_test'] == 'test']
    df_train = df_train.merge(df_train_label, how = 'left', on = 'TransactionID')

    df_train = df_train.drop(['train_or_test', 'TransactionID'], axis = 1)
    df_test = df_test.drop(['train_or_test', 'TransactionID'], axis = 1)
    print 'df_train.info() = ', df_train.info()
    return df_train, df_test

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

def train(df_train, df_test):
    # kFold_cross(df_train, 5, 'isFraud')
    label = np.array(df_train['isFraud'].tolist())
    train = df_train.drop(['isFraud'], axis = 1).values
    test = df_test.values
    xgb = xgboost.XGBRegressor()
    print type(train), type(label)
    print 'train.shape, test.shape, label.shape = ', train.shape, test.shape, label.shape
    xgb.fit(train, label)
    print 'set(label) = ', set(list(label))
    pred = xgb.predict(test)
    print type(pred)
    return pred

def test():
    df_train_tran = read_data(r'\train_transaction')
    train_cols = set(df_train_tran.columns.values)
    df_test_tran = read_data(r'\test_transaction')
    test_cols = set(df_test_tran.columns.values)
    print train_cols - test_cols

def main():
    time_now = time.time()
    df_train_glo, df_test_glo = read_all_data('all')
    # for ratio in [0.5, 0.4, 0.2, 0.1, 0.05]:
    for ratio in []:
        delete_cols = delete_null_feature(ratio)
        for drop_mask in ['fillna_mode', 'delete_row']:
            print '\n\ndelete_null_ratio = ', ratio, 'drop_mask = ', drop_mask, '-'*200
            df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, drop_mask)
            kFold_cross(df_train) # 0.97
            print time.time() - time_now
    delete_cols = delete_null_feature(0.1)
    df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, 'fillna_mode')
    pred = train(df_train, df_test)
    # write(pred)

if __name__ == '__main__':
    main()
    # test()


