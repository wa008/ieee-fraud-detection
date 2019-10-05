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


def data_preprocess(df_train_, df_test_, delete_cols = [], drop_mask = []):
    df_train = df_train_.copy()
    df_test = df_test_.copy()
    print 'data_preprocess', '-'*100
    print 'df_train, df_test.shape = ', df_train.shape, df_test.shape
    one_value_train = [col for col in df_train.columns if df_train[col].nunique() <= 1]
    one_value_test = [col for col in df_test.columns if df_test[col].nunique() <= 1]
    print one_value_train
    print one_value_test

def main():
    df_train_glo, df_test_glo = read_all_data('small')
    # df_train_glo, df_test_glo = read_all_data('all')
    data_preprocess(df_train_glo, df_test_glo)

if __name__ == '__main__':
    main()

