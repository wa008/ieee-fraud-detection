#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
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
    # df = pd.read_csv(data_path + data_name + '.csv', nrows = nrows)
    df = pd.read_csv(data_path + data_name + '.csv')
    print data_name[1:] + '_shape = ', df.shape, 'time = ', time.time() - now
    return df

def data_preprocess():
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
    print 'df_train, df_test, df shape = ', df_train.shape, df_test.shape, df.shape
    # 清楚缺失比例高的特征
    cols = df.columns.values
    delete_cols = []
    for col in tqdm(cols):
        x = sum(df[col].isna().values)
        null_ratio = x * 1.0 / len(df)
        if null_ratio > 0.5:
            delete_cols.append(col)
    df = df.drop(delete_cols, axis = 1)
    print 'after_delete_shape = ', df.shape
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

    df_train = df_train.drop(['train_or_test_train','train_or_test_test'], axis = 1)
    df_test = df_test.drop(['train_or_test_train','train_or_test_test'], axis = 1)
    return df_train, df_test

def train(df_train, df_test):
    kFold_cross(df_train, 5, 'isFraud')
    label = np.array(df_train['isFraud'].tolist())
    train = df_train.drop(['isFraud'], axis = 1).values
    test = df_test.values
    xgb = xgboost.XGBRegressor()
    print type(train), type(label)
    print train.shape, label.shape
    xgb.fit(train, label)
    print 'set(label) = ', set(list(label))
    pred = xgb.predict(test)
    print type(pred)
    write(pred)


def test():
    df_train_tran = read_data(r'\train_transaction')
    train_cols = set(df_train_tran.columns.values)
    df_test_tran = read_data(r'\test_transaction')
    test_cols = set(df_test_tran.columns.values)
    print train_cols - test_cols


def main():
    df_train, df_test = data_preprocess()
    kFold_cross(df_train) # 0.97
    # pred = train(df_train, df_test)
    # write(pred)

if __name__ == '__main__':
    main()
    # test()


