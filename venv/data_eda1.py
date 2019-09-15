#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from collections import Counter
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

# other file
from hello1 import read_all_data, read_middle_data, write_middle_data, delete_null_feature
from hello1 import card_fea, C_fea, D_fea, M_fea, addr_fea, ext_plus_fea, ext_reduce_fea, get_dummies_fea


def main():
    df_train, df_test = read_all_data('all')
    delete_fea = delete_null_feature(0.4)
    all_fea = df_train.columns.values
    use_fea = list((set(all_fea) - set(ext_reduce_fea) - set(delete_fea)) | set(get_dummies_fea) | set(ext_plus_fea))
    df_train_true = df_train[df_train['isFraud'] == 1]
    df_train_false = df_train[df_train['isFraud'] == 0]
    print 'df_train_true, df_train_false, df_train shape = ', df_train_true.shape, df_train_false.shape, df_train.shape
    mx_dis_play = 10000
    index = 0
    class_num = 0
    continity_num = 0
    for fea in tqdm(use_fea):
        index += 1
        if index > mx_dis_play:
            break
        data_true = df_train_true[fea].dropna().tolist()
        data_false = df_train_false[fea].dropna().tolist()
        data_true = Counter(data_true).items()
        data_false = Counter(data_false).items()
        data_true = np.array(sorted(data_true, key = lambda x:x[0]))
        data_false = np.array(sorted(data_false, key = lambda x:x[0]))
        print fea, 'shape = ', data_true.shape, data_false.shape
        if len(data_true) > 1 and len(data_false):
            plt.plot(data_true[:][0], data_true[:][1], c = 'r')
            plt.plot(data_false[:][0], data_false[:][1], c = 'b')
        else:
            print 'data_shape = 1, feature = ', fea, len(data_true), len(data_false), '-' * 100
        # plt.show()
        fea_type = 'continity'
        class_num += 1
        if fea in get_dummies_fea:
            fea_type = 'class'
            continity_num += 1
            class_num -= 1
        plt.title(fea_type + '_' + fea)
        plt.savefig(r'picture\pic_' + fea_type + r'Feature_' + fea + '.jpg')
        plt.close()
    print 'num = ', class_num, continity_num
    pass

main()
