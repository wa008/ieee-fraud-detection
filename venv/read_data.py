#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
# machine learning
# import xgboost
import warnings
warnings.filterwarnings("ignore")

data_path = r'D:\kaggle\data\ieee-fraud-detection'
now_path = r'D:\kaggle\ieee-fraud-detection\venv'
# print '中文'
def change_data(data_name):
    now = time.time()
    df_iden = pd.read_csv(data_path + data_name + '.csv')
    print 'time1 = ', time.time() - now
    print df_iden.shape
    df_iden.to_pickle(data_path + data_name + '.pkl')
    now = time.time()
    df_iden = pd.read_pickle(data_path +  data_name + '.pkl')
    print 'time2 = ',time.time() - now
    print df_iden.shape

def read_data_pkl(data_name):
    df = pd.read_pickle(data_path + data_name + '.pkl')
    return df

def read_data_csv(data_name, mark = 'all'):
    df = ''
    if mark == 'small':
        nrows = 500
        df = pd.read_csv(data_path + data_name + '.csv', nrows = nrows)
    elif mark == 'all':
        df = pd.read_csv(data_path + data_name + '.csv')
    return df

def read_all_data(mark = 'small'):
    print 'read_all_data', '-'*100
    now = time.time()
    if mark == 'small':
        df_train_tran = read_data_csv(r'\train_transaction', mark)
        df_train_iden = read_data_csv(r'\train_identity', mark)
        df_test_tran = read_data_csv(r'\test_transaction', mark)
        df_test_iden = read_data_csv(r'\test_identity', mark)
    else:
        df_train_tran = read_data_pkl(r'\train_transaction')
        df_train_iden = read_data_pkl(r'\train_identity')
        df_test_tran = read_data_pkl(r'\test_transaction')
        df_test_iden = read_data_pkl(r'\test_identity')
    df_train = df_train_tran.merge(df_train_iden, how = 'left', on = 'TransactionID')
    df_test = df_test_tran.merge(df_test_iden, how = 'left', on = 'TransactionID')
    del df_train_tran, df_train_iden, df_test_tran, df_test_iden
    print 'read_data_time = ', time.time() - now
    return df_train, df_test

def write_middle_data(data, data_name):
    f = open(now_path + data_name, 'w')
    f.write(data)
    f.close()

def read_middle_data(data_name):
    f = open(now_path + data_name, 'r')
    return f.read()

def main():
    # read_data_csv(r'\train_transaction')
    # read_data_pkl(r'\train_transaction')
    # change_data(r'\train_transaction')
    # change_data(r'\train_identity')
    # change_data(r'\test_transaction')
    # change_data(r'\test_identity')
    # change_data(r'\sample_submission')
    pass

if __name__ == '__main__':
    main()
