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

def main():
    data_size = sys.argv[1] # all or small
    df_train, df_test = read_all_data(data_size)
    print df_train.shape, df_test.shape

    row = 3
    col = 5
    for i in range(1, 16):
        plt.subplot(row, col, i)
        plt.scatter(df_train['TransactionDT'], df_train['D' + str(i)])
        plt.title('D' + str(i))

    plt.savefig('./picture/Transaction_D_index.jpg')
    plt.show()

def plot1():
    data_size = sys.argv[1] # all or small
    df_train, df_test = read_all_data(data_size)

    plt.figure(figsize=(15,5))
    plt.scatter(df_train.TransactionDT,df_train.D15)
    plt.title('Original D15')
    plt.xlabel('Time')
    plt.ylabel('D15')
    plt.show()


main()
# plot1()