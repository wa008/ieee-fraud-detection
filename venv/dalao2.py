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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit, KFold
import xgboost
# myself libs
from kFold_corss import kFold_cross
from write_result import write
from add_other_feature1 import add_other_feature
from read_data import read_all_data
from read_data import read_middle_data
from train import train
# from dalao_func1 import train_model_classification
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

card_fea = ['card' + str(i) for i in range(1, 7)]
C_fea = ['C' + str(i) for i in range(1, 15)]
D_fea = ['D' + str(i) for i in range(1, 16)]
M_fea = ['M' + str(i) for i in range(1, 10)]
addr_fea = ['addr' + str(i) for i in range(1, 3)]

ext_reduce_fea = []
get_dummies_fea = ['ProductCD', 'P_emaildomain', 'R_emaildomain'] + card_fea + addr_fea + M_fea + ['DeviceType']
ext_plus_fea = ['P_emaildomain'] + M_fea + addr_fea
ext_plus_fea = [x for x in get_dummies_fea]


def data_preprocess(df_train_, df_test_, delete_cols = [], drop_mask = []):
    train = df_train_.copy()
    test = df_test_.copy()

    train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
    train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
    train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
    train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

    test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
    test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
    test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
    test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

    train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
    train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
    train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
    train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

    test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
    test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
    test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
    test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

    train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
    train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
    train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
    train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

    test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
    test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
    test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
    test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

    train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
    train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
    train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
    train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')

    test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
    test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')
    test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
    test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')

    train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
    train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
    test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
    test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)
    many_null_cols = [col for col in train.columns if train[col].isna().sum() * 1.0 / train.shape[0] > 0.9]
    many_null_cols_test = [col for col in test.columns if test[col].isna().sum() * 1.0 / test.shape[0] > 0.9]
    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna = False, normalize = True).values[0] > 0.9]
    big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna = False, normalize = True).values[0] > 0.9]
    one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
    one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
    cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test +
                            one_value_cols + one_value_cols_test))
    cols_to_drop.remove('isFraud')
    print len(cols_to_drop)
    train = train.drop(cols_to_drop, axis = 1)
    test = test.drop(cols_to_drop, axis = 1)
    cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
    for col in cat_cols:
        if col in train.columns:
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))
            test[col] = le.transform(list(test[col].astype(str).values))
    X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y = train.sort_values('TransactionDT')['isFraud']
    X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
    del train
    test = test[["TransactionDT", 'TransactionID']]
    # by https://www.kaggle.com/dimartinot
    def clean_inf_nan(df):
        return df.replace([np.inf, -np.inf], np.nan)

    # Cleaning infinite values to NaN
    X = clean_inf_nan(X)
    X_test = clean_inf_nan(X_test)
    return X, y, X_test

def train_dalao(X, y, X_test):
    n_fold = 5
    folds = TimeSeriesSplit(n_splits=n_fold)
    folds = KFold(n_splits=5)
    params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }
    result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,
                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual', n_jobs=-1)
    return result_dict_lgb['prediction']

def train_lgb(X, y, X_test):
    model = lgb.LGBMRegressor()
    model.fit(X, y, eval_metric='AUC')
    ypred = model.predict(X_test)
    ypred = np.maximum(0, ypred)
    ypred = np.minimum(1, ypred)
    return ypred

def train_xgb(X, y, X_test):
    model = xgboost.XGBRegressor()
    model.fit(X, y, eval_metric='auc')
    ypred = model.predict(X_test)
    ypred = np.maximum(0, ypred)
    ypred = np.minimum(1, ypred)
    return ypred

def main():
    df_train_glo, df_test_glo = read_all_data('all')
    # df_train_glo, df_test_glo = read_all_data('small')
    X, y, X_test = data_preprocess(df_train_glo, df_test_glo)
    # ypred = train_lgb(X, y, X_test)
    # write(ypred, '1005_2')
    ypred = train_xgb(X, y, X_test)
    write(ypred, '1005_2')

if __name__ == '__main__':
    main()

# labelencode, gc.collect, P_mesxxxxxx
