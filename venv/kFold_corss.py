#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm
# machine learning
import xgboost
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'
random.seed(2019)

def kFold_cross(df_train, k = 5, label = 'isFraud'):
    # df_train = df_train.sample(frac = 1.0)
    df_train = df_train.sort_values(by = ['TransactionDT'])
    leng = len(df_train) / k
    index = (np.array(range(k)) * leng).tolist()
    index.append(len(df_train))
    print 'index = ', index
    ratio = 0.0
    for i in tqdm(range(k)):
        df_test_tmp = df_train[index[i] : index[i+1]][:].reset_index()
        df_train_tmp = pd.concat([df_train[: index[i]][:], df_train[index[i+1] : ][:]]).reset_index()
        y_train = df_train_tmp[label].astype(np.float).tolist()
        df_train_tmp = df_train_tmp.drop([label], axis = 1)
        y_true = df_test_tmp[label].astype(np.float).tolist()
        df_test_tmp = df_test_tmp.drop([label], axis = 1)
        xgb = xgboost.XGBRegressor()
        xgb.fit(df_train_tmp, y_train)
        y_pred = list(xgb.predict(df_test_tmp))
        y_pred = np.maximum(y_pred, 0)
        y_pred = np.minimum(y_pred, 1)
        # for i in range(len(y_pred)):
        #     if y_pred[i] < 0:
        #         y_pred[i] = 0
        #     elif y_pred[i] > 1:
        #         y_pred[i] = 1
        print type(y_pred), type(y_true)
        print len(y_pred), len(y_true)
        ratio += 1 - np.sum(np.array(y_pred) - np.array(y_true)) * 1.0 / len(df_test_tmp)
    print str(k) + '_Flod_corss_accracy = ', ratio / k
    return ratio / k

