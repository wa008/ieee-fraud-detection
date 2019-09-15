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

from hello1 import read_all_data
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
    # df_tran = pd.read_csv(data_path + r'\train_transaction.csv')
    print df_iden.shape

def read_data_pkl(data_name):
    now = time.time()
    df = pd.read_pickle(data_path + data_name + '.pkl')
    print data_name[1:] + '_shape = ', df.shape, 'time = ', time.time() - now
    return df

def read_data_csv(data_name):
    now = time.time()
    df = pd.read_csv(data_path + data_name + '.csv')
    print data_name[1:] + '_shape = ', df.shape, 'time = ', time.time() - now
    return df

def main():
    df_train, df_test = read_all_data('small')
    df_train.to_csv(data_path + r'\train_small.csv', index = False)
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
