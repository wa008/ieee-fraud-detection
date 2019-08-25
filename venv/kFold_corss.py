#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
# machine learning
import xgboost
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'
random.seed(2019)

def kFold_cross(df_train, k = 5, label = 'isFraud'):
    df_train = df_train.sample(frac = 1.0)
    leng = len(df_train) / k
    index = (np.array(range(k)) * leng).tolist()
    index.append(len(df_train))
    print 'index = ', index
    ratio = 0.0
    for i in range(k):
        df_test_tmp = df_train[index[i] : index[i+1]][:].reset_index()
        df_train_tmp = pd.concat([df_train[: index[i]][:], df_train[index[i+1] : ][:]]).reset_index()
        y_train = df_train_tmp[label].tolist()
        df_train_tmp = df_train_tmp.drop([label], axis = 1)
        y_true = df_test_tmp[label].astype(int).tolist()
        df_test_tmp = df_test_tmp.drop([label], axis = 1)
        xgb = xgboost.XGBRegressor()
        xgb.fit(df_train_tmp, y_train)
        y_pred = xgb.predict(df_test_tmp)
        y_pred = [1 if x>=0.5 else 0 for x in y_pred]
        print type(y_pred), type(y_true)
        print len(y_pred), len(y_true)
        ratio += sum(np.array(y_pred) == np.array(y_true)) * 1.0 / len(df_test_tmp)
    print str(k) + '_Flod_corss_accracy = ', ratio / k
    return ratio / k

