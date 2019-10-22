#coding=utf-8
import numpy as np
import pandas as pd
import math
import random
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# import xgboost
# machine learning
# import xgboost
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'

df = pd.DataFrame([
    [1,2],
    [1,3],
    [3,2]
], columns=['a', 'b'])

def main_test():
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

def ratio_dict_test():
    pass
    d = json.loads(read_middle_data('null_ratio_dict.txt'))
    eps = 0.05
    ans = [0] * int(1.0 / eps)
    for x, y in d.items():
        ans[int(y / eps)] += 1
    for i in range(int(1.0 / eps)):
        print i*eps, '-', i*eps + eps, ans[i]

def test1():
    df = pd.DataFrame([
        [1, 'a'],
        [1, 'b'],
        [1, np.nan],
    ], columns=['a', 'b'])
    print df
    df['a'] = df['a'].astype(np.str)
    print df.info()
    df['b'] = df['b'].fillna(df['b'].mode())
    print df['b'].dtype == 'object'
    print df

def test2():
    x = []
    print np.min(x)

def test3():
    x = 1
    y = 21212.2323
    print '%10.3f %10.3f' % (x, x)
    print '%10.3f %10.3f' % (y, y)
    xx = np.array([1,2,3])
    print type(np.min(xx))
    print '%10.3f %10.3f' % (float(np.min(xx)), y)

def test4():
    x = [1, 2, 1, 2, 3]
    plt.hist(x)
    plt.show()

def test5(x):
    print 'test5_param = ', x
    def test6(x):
        print 'test6_param = ', x
        return x * x
    return test6

def test6():
    def f(x):
        return {'loss' : x**2, 'status' : STATUS_OK}
    t = Trials()
    best = fmin(
        fn=f,
        space=hp.uniform('x', -200, 20),
        algo=tpe.suggest,
        max_evals=100)
    print best
    for x in t.trials[: 10]:
        print x
def main():
    # ratio_dict_test()
    # test1()
    # test2()
    # test3()
    # test4()
    # test6()
    pass
# main()
# test1()
# model_test()
# test2()
# data_test()
# random_test1()

main()
