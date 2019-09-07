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

ext_plus_fea = ['P_emaildomain']
ext_reduce_fea = []

def data_preprocess(delete_cols):
    now = time.time()
    df_train_tran = read_data(r'\train_transaction')
    df_train_iden = read_data(r'\train_identity')
    df_test_tran = read_data(r'\test_transaction')
    df_test_iden = read_data(r'\test_identity')
    print 'read_data_time = ', time.time() - now
    # print df_train_tran.info()
    df_train = df_train_tran.merge(df_train_iden, how = 'left', on = 'TransactionID')
    df_test = df_test_tran.merge(df_test_iden, how = 'left', on = 'TransactionID')
    df_train = df_train[~df_train['isFraud'].isna()]
    df_train['train_or_test'] = np.array(['train'] * len(df_train))
    df_test['train_or_test'] = np.array(['test'] * len(df_test))
    df_train_label = df_train[['TransactionID', 'isFraud']]
    df_train = df_train.drop(['isFraud'], axis = 1)
    df = pd.concat([df_train, df_test])

    df_copy = df.copy()
    df = df.drop(delete_cols, axis = 1)
    for fea in ext_plus_fea:
        df[fea] = df_copy[fea]
    df = df.drop(ext_reduce_fea, axis = 1)
    print 'after_delete_plut_reduct_shape = ', df.shape
    print 'df_train, df_test, df shape = ', df_train.shape, df_test.shape, df.shape
    # 特征类型转换
    cols = df.columns.values
    str_col_num = 0
    for col in tqdm(cols):
        try:
            df[col] = df[col].astype(np.float)
        except:
            str_col_num += 1
            pass
    df = df.fillna(-1)
    df = pd.get_dummies(df)
    print 'after ont-hot = ', df.shape, 'str_col_num = ', str_col_num
    df_train = df[df['train_or_test_train'] == 1]
    df_test = df[df['train_or_test_test'] == 1]
    df_train = df_train.merge(df_train_label, how = 'left', on = 'TransactionID')

    df_train = df_train.drop(['train_or_test_train','train_or_test_test', 'TransactionID'], axis = 1)
    df_test = df_test.drop(['train_or_test_train','train_or_test_test', 'TransactionID'], axis = 1)
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
    # for ratio in [0.4, 0.2, 0.1, 0.05]:
    for ratio in []:
        print '\ndelete_null_ratio = ', ratio
        delete_cols = delete_null_feature(ratio)
        df_train, df_test = data_preprocess(delete_cols)
        kFold_cross(df_train) # 0.97
        print time.time() - time_now
    delete_cols = delete_null_feature(0.1)
    df_train, df_test = data_preprocess(delete_cols)
    pred = train(df_train, df_test)
    write(pred)

if __name__ == '__main__':
    main()
    # test()


