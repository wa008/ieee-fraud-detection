#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
from tqdm import tqdm
# machine learning
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from collections import Counter
import xgboost
# myself libs
from kFold_corss import kFold_cross
from write_result import write
from add_other_feature1 import add_other_feature
from read_data import read_all_data
from read_data import read_middle_data
from train import train
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
from xgboost import plot_importance
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import KFold,TimeSeriesSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import gc

card_fea = ['card' + str(i) for i in range(1, 7)]
C_fea = ['C' + str(i) for i in range(1, 15)]
D_fea = ['D' + str(i) for i in range(1, 16)]
M_fea = ['M' + str(i) for i in range(1, 10)]
addr_fea = ['addr' + str(i) for i in range(1, 3)]

ext_reduce_fea = []
get_dummies_fea = ['ProductCD', 'P_emaildomain', 'R_emaildomain'] + card_fea + addr_fea + M_fea + ['DeviceType']
ext_plus_fea = ['P_emaildomain'] + M_fea + addr_fea
ext_plus_fea = [x for x in get_dummies_fea]

def delete_null_feature(ratio = 0.4):
    # 清除缺失比例高的特征
    delete_cols = []
    null_ratio_dict = json.loads(read_middle_data(r'\null_ratio_dict.txt'))
    cols = null_ratio_dict.keys()
    for col in tqdm(cols):
        # x = sum(df[col].isna().values)
        # null_ratio = x * 1.0 / len(df)
        null_ratio = null_ratio_dict[col]
        # null_ratio_dict[col] = null_ratio
        if null_ratio > ratio:
            delete_cols.append(col)
    return delete_cols
    # write_middle_data(json.dumps(null_ratio_dict), 'null_ratio_dict.txt')

def delete_both_feature(ratio = 0.4):
    lines = list(read_middle_data(r'\both_single_num_sort.txt').split('\n'))[:-3]
    all_list = []
    for line in lines:
        col = line.split(' ')[0]
        all_list.append(col)
    all_len = len(all_list)
    return all_list[int(all_len * ratio):]

def data_preprocess(df_train_, df_test_, delete_cols, drop_mask):
    df_train = df_train_.copy()
    df_test = df_test_.copy()
    print 'data_preprocess', '-'*100
    print 'df_train, df_test.shape = ', df_train.shape, df_test.shape
    # print df_train.columns.values
    df_train = df_train[~df_train['isFraud'].isna()]
    df_train['train_or_test'] = np.array(['train'] * len(df_train))
    df_test['train_or_test'] = np.array(['test'] * len(df_test))
    df_train_label = df_train[['TransactionID', 'isFraud']]
    df_train = df_train.drop(['isFraud'], axis = 1)
    df = pd.concat([df_train, df_test])
    df = add_other_feature(df)

    print 'in P_emaildomain' in get_dummies_fea
	# 特征增删
    df = df.drop(list(set(delete_cols + ext_reduce_fea) - set(ext_plus_fea)), axis = 1)
    print 'after_delete_plus_reduce_shape = ', df.shape
    print 'df_train, df_test, df shape = ', df_train.shape, df_test.shape, df.shape

    # 缺失值填充
    for col in get_dummies_fea:
        df[col] = df[col].astype(np.str)
    df_train = df[df['train_or_test'] == 'train']
    df_test = df[df['train_or_test'] == 'test']
    cols = df_train.columns.values
    print 'before drop_mask df_train = ', df_train.shape
    if drop_mask == 'fillna_mode':
        for col in tqdm(cols):
            if df_train[col].dtype == 'object':
                df_train[col] = df_train[col].fillna(df_train[col].mode())
            else:
                df_train[col] = df_train[col].fillna(df_train[col].mean())
    elif drop_mask == 'delete_row':
        df_train = df_train.dropna(axis=0)
    print 'after drop_mask df_train = ', df_train.shape
    cols = df_test.columns.values
    for col in tqdm(cols):
        if df_test[col].dtype == 'object':
            df_test[col] = df_test[col].fillna(df_test[col].mode())
        else:
            df_test[col] = df_test[col].fillna(df_test[col].mean())
    df = pd.concat([df_train, df_test])
    print 'in P_emaildomain' in get_dummies_fea

    # one-hot
    print 'one-hot', '-' * 100
    get_dummies_fea_copy = [x for x in get_dummies_fea]
    temp = get_dummies_fea
    for col in temp:
        mid = len(set(df[col].tolist()))
        if mid > 15:
            df = df.drop([col], axis = 1)
            get_dummies_fea_copy.remove(col)
            print 'drop feature = ', col, mid
    print 'before ont-hot df.shape = ', df.shape
    oneHotEncode = OneHotEncoder(sparse=False)
    df_tmp = pd.DataFrame(oneHotEncode.fit_transform(df[get_dummies_fea_copy])).reset_index()
    df = df.reset_index()
    df = df.drop(get_dummies_fea_copy + ['index'], axis = 1)
    df = pd.concat([df, df_tmp], axis = 1)
    cols = df.columns.values
    for col in tqdm(cols):
        df[col] = df[col].astype(np.int, errors = 'ignore')
        df[col] = df[col].astype(np.float, errors = 'ignore')
        if df[col].dtypes == 'object':
            print 'null float col = ', col
    print 'after ont-hot df.shape = ', df.shape

    df_train = df[df['train_or_test'] == 'train']
    df_test = df[df['train_or_test'] == 'test']
    df_train = df_train.merge(df_train_label, how = 'left', on = 'TransactionID')

    df_train = df_train.drop(['train_or_test', 'TransactionID'], axis = 1)
    df_test = df_test.drop(['train_or_test', 'TransactionID'], axis = 1)
    return df_train, df_test

def objective(params):
    time1 = time.time()
    params = {
        'max_depth': int(params['max_depth']),
        # 'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.0f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.0f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }
    print("\n############## New Run ################")
    print 'params = ', params
    FOLDS = 7
    count=1
    # skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    tss = TimeSeriesSplit(n_splits=FOLDS)
    score_mean = 0
    for tr_idx, val_idx in tss.split(X_train, y_train):
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt',
            n_estimators=600,
            **params
        )
        print 'tr_idx, val_idx = '
        print type(tr_idx), len(tr_idx)
        print type(val_idx), len(val_idx)
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        # plt.show()
        score_mean += score
        print count, 'CV - score:', round(score, 4),
        count += 1
    time2 = time.time() - time1
    print 'Total Time Run: ', round(time2 * 1.0 / 60,2)
    gc.collect()
    print 'Mean ROC_AUC: ', score_mean * 1.0 / FOLDS
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

def train_lgb(X, y, X_test, params={}):
    print 'train_xgb', '-' * 100
    global X_train, y_train
    X_train = X
    y_train = y
    space = {
        'max_depth': hp.quniform('max_depth', 7, 23, 1),
        'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),
        # 'gamma': hp.uniform('gamma', 0.01, .7),
        'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
        'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
        'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
        'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)
    }
    best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=27)
    best_params = space_eval(space, best)
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    model = lgb.LGBMClassifier(n_estimators = 600, **best_params)
    model.fit(X, y)
    ypred = model.predict_proba(X_test)[:, 1]
    return ypred

def main():
    time_now = time.time()
    # df_train_glo, df_test_glo = read_all_data('small')
    df_train_glo, df_test_glo = read_all_data('all')
    global get_dummies_fea
    must_delete = delete_null_feature(0.4)

    best_ratio = 0.8
    best_mask = 'delete_row'
    delete_cols = delete_both_feature(best_ratio) + must_delete
    df_train, df_test = data_preprocess(df_train_glo, df_test_glo, delete_cols, best_mask)
    print df_train.columns.values
    print df_test.columns.values
    y = df_train['isFraud']
    X = df_train.drop(['isFraud', 'TransactionDT'], axis = 1)
    X_test = df_test.drop(['TransactionDT'], axis = 1)
    y_pred = train_lgb(X, y, X_test)
    write(y_pred, '1008_1')

if __name__ == '__main__':
    main()
    # test()

