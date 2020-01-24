#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
import sys
from tqdm import tqdm
# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from collections import Counter
import matplotlib.pyplot as plt
import xgboost
sys.path.append('..')
# myself libs
from read_data import read_all_data
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

df_train, df_test = read_all_data('all')
print df_train.shape, df_test.shape
print df_train.info()

cols = []
for col in df_train.columns:
    if 'D' in col:
        print col
        cols.append(col)

print df_train[cols].describe()

row = 5
col = 3
for i in range(1, 16):
    x = np.sum(df_train['D' + str(i)].isna())
    print i, x, x * 1.0 / len(df_train)
    plt.subplot(row, col, i)
    plt.hist(df_train['D' + str(i)].dropna(), bins = 10)
plt.show()


