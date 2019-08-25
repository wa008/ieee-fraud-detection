#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import xgboost
# machine learning
# import xgboost
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'

def read_data(data_name):
    now = time.time()
    df = pd.read_pickle(data_path + data_name + '.pkl')
    print data_name[1:] + '_shape = ', df.shape, 'time = ', time.time() - now
    return df

df = pd.DataFrame([
    [1,2],
    [2,3],
    [3,4]
], columns=['a', 'b'])

def main():
    global df
    df1 = pd.DataFrame([
        [1,'a', 2],
        [2,3, '12'],
        ['10', '11','12'],
        [4,4,'12']
    ], columns=['a', 'c', 'b'])
    try:
        df1 = df1.astype(np.float)
    except:
        pass
    print df1
    print df1.info()
main()
def model_test():
    xgb = xgboost.XGBRegressor()
    x_train = np.array([['a',2], ['b',3], ['c', 3.0]])
    df_train = pd.DataFrame(x_train, columns = ['a', 'b'])
    for col in df_train.columns.values:
        try:
            df_train[col] = df_train[col].astype(np.int)
        except:
            try:
                df_train[col] = df_train[col].astype(np.float)
            except:
                pass
    print df_train.info()
    df_train = pd.get_dummies(df_train)
    print df_train
    return
    x_train = df_train.values
    y_train = np.array([1,2])
    xgb.fit(x_train, y_train)
    pred = xgb.predict(np.array([[1,2], [2,3]]))
    print pred
def test1():
    df_train_tran = read_data(r'\train_transaction')
    train_cols = sorted(set(df_train_tran.columns.values), key = lambda x:[x[0:1], int(x[1:]) if(str(x[1:]).isdigit()) else str(x[1:])])
    print train_cols
    return 1
    df_test_tran = read_data(r'\test_transaction')
    test_cols = set(df_test_tran.columns.values)
    print train_cols - test_cols

def test2():
    global df

def data_test():
    x = range(1000)
    s = 0
    for i in tqdm(x, total=len(x)):
        time.sleep(0.01)
        s += i
    print s
def random_test1():
    # random.seed(2019)
    # index = range(10)
    # random.shuffle(index)
    # print index
    global df
    df = df.sample(frac = 1.0).reset_index()
    print df

# main()
# test1()
# model_test()
# test2()
# data_test()
# random_test1()
