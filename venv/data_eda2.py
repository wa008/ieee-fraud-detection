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
    # df_train, df_test = read_all_data('small')
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
    both_num = []
    for fea in tqdm(use_fea):
        index += 1
        if index > mx_dis_play:
            break
        data_true = sorted(df_train_true[fea].dropna().tolist())
        data_false = sorted(df_train_false[fea].dropna().tolist())
        true_single_list = []
        false_single_list = []
        both_single_list = []
        i1 = 0
        i2 = 0
        while i1 < len(data_true) and i2 < len(data_false):
            if data_true[i1] == data_false[i2]:
                both_single_list.append(data_true[i1])
                i1 += 1
                i2 += 1
            elif data_true[i1] < data_false[i2]:
                true_single_list.append(data_true[i1])
                i1 += 1
            else:
                false_single_list.append(data_false[i2])
                i2 += 1
        while i1 < len(data_true):
            true_single_list.append(data_true[i1])
            i1 += 1
        while i2 < len(data_false):
            false_single_list.append(data_false[i2])
            i2 += 1
        print 'fea =', fea, 'type =', df_train[fea].dtype
        both_num.append([fea, len(both_single_list)])
        print len(true_single_list), len(false_single_list), len(both_single_list)
        if df_train[fea].dtype != 'object':
            print 'max, min, ave, std = '
            if len(true_single_list) > 0:
                # print '%10.3f %10.3f %10.3f' % (np.max(true_single_list), np.min(true_single_list), np.average(true_single_list), np.std(true_single_list))
                print np.max(true_single_list), np.min(true_single_list), np.average(true_single_list), np.std(true_single_list)
            if len(false_single_list) > 0:
                print np.max(false_single_list), np.min(false_single_list), np.average(false_single_list), np.std(false_single_list)
                # print '%10.3f %10.3f %10.3f' % (np.max(false_single_list), np.min(false_single_list), np.average(false_single_list), np.std(false_single_list))
            if len(both_single_list) > 0:
                # print '%10.3f %10.3f %10.3f' % (np.max(both_single_list), np.min(both_single_list), np.average(both_single_list), np.std(both_single_list))
                print np.max(both_single_list), np.min(both_single_list), np.average(both_single_list), np.std(both_single_list)
        print '\n'
    both_num = sorted(both_num, key = lambda x:x[1])
    for x in both_num:
        print x[0], x[1]
    both_num = [str(x[0]) + ' ' + str(x[1]) for x in both_num]
    both_num.append(str(len(df_train_true)))
    both_num.append(str(len(df_train_false)))
    both_num.append(str(len(df_train)))
    write_middle_data('\n'.join(both_num), 'both_single_num_sort.txt')
    print 'num = ', class_num, continity_num
    pass

main()
