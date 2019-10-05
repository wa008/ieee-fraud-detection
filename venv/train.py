#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
from tqdm import tqdm
import xgboost
# other
import warnings
warnings.filterwarnings("ignore")

def train(df_train, df_test):
    # kFold_cross(df_train, 5, 'isFraud')
    label = np.array(df_train['isFraud'].tolist())
    train = df_train.drop(['isFraud'], axis = 1).values
    test = df_test.values
    xgb = xgboost.XGBRegressor()
    print 'train.shape, test.shape, label.shape = ', train.shape, test.shape, label.shape
    xgb.fit(train, label)
    pred = xgb.predict(test)
    return pred