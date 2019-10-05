#coding=utf-8
import numpy as np
import pandas as pd
import random
import math
import time
from tqdm import tqdm
# machine learning
import warnings
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'
random.seed(2019)

def kFold_cross(df_train, model_begin = xgboost.XGBRegressor(), k = 5, label = 'isFraud'):
    # df_train = df_train.sample(frac = 1.0)
    print 'kFold_cross', '-'*100
    print 'df.shape = ', df_train.shape
    if len(df_train) < 5:
        return -1, -1
    obj_cols = (df_train.dtypes == 'object')
    if len(obj_cols) > 0:
        print obj_cols[obj_cols].index
    df_train = df_train.sort_values(by = ['TransactionDT'])
    leng = len(df_train) / k
    index = (np.array(range(k)) * leng).tolist()
    index.append(len(df_train))
    print 'index = ', index
    ratio = 0.0
    loss = 0
    mse = 0
    for i in tqdm(range(k)):
        df_test_tmp = df_train[index[i] : index[i+1]][:].reset_index()
        df_train_tmp = pd.concat([df_train[: index[i]][:], df_train[index[i+1] : ][:]]).reset_index()
        y_train = df_train_tmp[label].astype(np.float).tolist()
        df_train_tmp = df_train_tmp.drop([label], axis = 1)
        y_true = df_test_tmp[label].astype(np.float).tolist()
        df_test_tmp = df_test_tmp.drop([label], axis = 1)
        model = xgboost.XGBRegressor()
        model.fit(df_train_tmp.values, y_train)
        y_pred = list(model.predict(df_test_tmp.values))
        y_pred = np.maximum(y_pred, 0.0000001)
        y_pred = np.minimum(y_pred, 0.9999999)
        for j in range(len(list(y_pred))):
            loss += - y_true[j] * math.log(y_pred[j]) - (1 - y_true[j]) * math.log(1 - y_pred[j])
            mse += mean_absolute_error(y_true, y_pred)
        ratio += 1 - np.sum(np.array(y_pred) - np.array(y_true)) * 1.0 / len(df_test_tmp)
    print str(k) + '_Flod_corss_accracy = ', ratio / k, 'lr_loss = ', loss, 'mse = ', mse
    return loss, mse

