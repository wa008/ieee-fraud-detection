#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
import sys
import gc
from tqdm import tqdm
# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from collections import Counter
import xgboost
import datetime
sys.path.append('..')
# myself libs
from read_data import read_all_data
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

data_path = r'D:\kaggle\data\ieee-fraud-detection\after_compete\p'[: -1]
print data_path

def plot():
    # PLOT ORIGINAL D
    plt.figure(figsize=(15,5))
    plt.scatter(X_train.TransactionDT,X_train.D15)
    plt.title('Original D15')
    plt.xlabel('Time')
    plt.ylabel('D15')
    plt.show()
    # PLOT TRANSFORMED D
    plt.figure(figsize=(15,5))
    plt.scatter(X_train.TransactionDT,X_train.D15)
    plt.title('Transformed D15')
    plt.xlabel('Time')
    plt.ylabel('D15n')
    plt.show()


def data_normalize(X_train, X_test):
    # NORMALIZE D COLUMNS
    for i in range(1,16):
        if i in [1,2,3,5,9]: continue
        X_train['D'+str(i)] =  X_train['D'+str(i)] - X_train.TransactionDT/np.float32(24*60*60)
        X_test['D'+str(i)] = X_test['D'+str(i)] - X_test.TransactionDT/np.float32(24*60*60) 
    return X_train, X_test


# FREQUENCY ENCODE TOGETHER
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        
# LABEL ENCODE
def encode_LE(col, train, test, verbose = True):
    df_comb = pd.concat([train[col],test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max()>32000: 
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb; x=gc.collect()
    if verbose: 
        print 'nm :', nm
        
# GROUP AGGREGATION MEAN AND STD
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
def encode_AG(main_columns, uids, aggregations=['mean'], train_df = 1, test_df = 1,
              fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                print new_col_name
                
# COMBINE FEATURES
def encode_CB(col1,col2,df1 = 1,df2 = 1):
    nm = col1+'_'+col2
    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 
    encode_LE(nm, df1, df2, verbose=False)
    print 'nm :', nm
    
# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, train_df = 1, test_df = 1):
    for main_column in main_columns:  
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            print new_col_name

def encode(X_train, X_test):
    # TRANSACTION AMT CENTS
    X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
    X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
    print 'cents, '
    # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
    encode_FE(X_train,X_test,['addr1','card1','card2','card3','P_emaildomain'])
    # COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN
    encode_CB('card1','addr1', X_train, X_test)
    encode_CB('card1_addr1','P_emaildomain', X_train, X_test)
    # FREQUENCY ENOCDE
    encode_FE(X_train,X_test,['card1_addr1','card1_addr1_P_emaildomain'])
    # GROUP AGGREGATE
    encode_AG(['TransactionAmt','D9','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'], X_train, X_test, usena=True)
    return X_train, X_test

def remove_cols(X_train, X_test = -1):
    cols = list( X_train.columns )
    cols.remove('TransactionDT')
    for c in ['D6','D7','D8','D9','D12','D13','D14']:
        cols.remove(c)
        
    # FAILED TIME CONSISTENCY TEST
    for c in ['C3','M5','id_08','id_33']:
        cols.remove(c)
    for c in ['card4','id_07','id_14','id_21','id_30','id_32','id_34']:
        cols.remove(c)
    for c in ['id_'+str(x) for x in range(22,28)]:
        cols.remove(c)
    return cols

def train95():
    print 'NOW USING THE FOLLOWING',len(cols),'FEATURES.'
    np.array(cols)

    # CHRIS - TRAIN 75% PREDICT 25%
    idxT = X_train.index[:3*len(X_train)//4]
    idxV = X_train.index[3*len(X_train)//4:]
    # KONSTANTIN - TRAIN 4 SKIP 1 PREDICT 1 MONTH
    #idxT = X_train.index[:417559]
    #idxV = X_train.index[-89326:]

import xgboost as xgb
print "XGBoost version:", xgb.__version__

def BUILD95():
    clf = xgb.XGBClassifier( 
        n_estimators=2000,
        max_depth=12, 
        learning_rate=0.02, 
        subsample=0.8,
        colsample_bytree=0.4, 
        missing=-1, 
        eval_metric='auc',
        # USE CPU
        #nthread=4,
        #tree_method='hist' 
        # USE GPU
        tree_method='gpu_hist' 
    )
    h = clf.fit(X_train.loc[idxT,cols], y_train[idxT], 
        eval_set=[(X_train.loc[idxV,cols],y_train[idxV])],
        verbose=50, early_stopping_rounds=100)

def build95_feature_imp():
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cols)), columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:50])
    plt.title('XGB95 Most Important Features')
    plt.tight_layout()
    plt.show()
    del clf, h; x=gc.collect()

def other2():
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    X_train['DT_M'] = X_train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    X_train['DT_M'] = (X_train['DT_M'].dt.year-2017)*12 + X_train['DT_M'].dt.month 

    X_test['DT_M'] = X_test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    X_test['DT_M'] = (X_test['DT_M'].dt.year-2017)*12 + X_test['DT_M'].dt.month 

def BUILD95_train():
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    skf = GroupKFold(n_splits=6)
    for i, (idxT, idxV) in enumerate( skf.split(X_train, y_train, groups=X_train['DT_M']) ):
        month = X_train.iloc[idxV]['DT_M'].iloc[0]
        print 'Fold',i,'withholding month',month
        print ' rows of train =',len(idxT),'rows of holdout =',len(idxV) 
        print col, main_column
        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric='auc',
            # USE CPU
            #nthread=4,
            #tree_method='hist'
            # USE GPU
            tree_method='gpu_hist' 
        )        
        h = clf.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT], 
                eval_set=[(X_train[cols].iloc[idxV],y_train.iloc[idxV])],
                verbose=100, early_stopping_rounds=200)
    
        oof[idxV] += clf.predict_proba(X_train[cols].iloc[idxV])[:,1]
        preds += clf.predict_proba(X_test[cols])[:,1]/skf.n_splits
        del h, clf
        x=gc.collect()
    print '#'*20
    print 'XGB95 OOF CV=',roc_auc_score(y_train,oof)


def BUILD95_plot():
    plt.hist(oof,bins=100)
    plt.ylim((0,5000))
    plt.title('XGB OOF')
    plt.show()

    X_train['oof'] = oof
    X_train.reset_index(inplace=True)
    X_train[['TransactionID','oof']].to_csv('oof_xgb_95.csv')
    X_train.set_index('TransactionID',drop=True,inplace=True)


def BUILD95_output_and_plot():
    sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
    sample_submission.isFraud = preds
    sample_submission.to_csv('sub_xgb_95.csv',index=False)

    plt.hist(sample_submission.isFraud,bins=100)
    plt.ylim((0,5000))
    plt.title('XGB95 Submission')
    plt.show()

def get_uid(X_train, X_test):
    X_train['day'] = X_train.TransactionDT / (24*60*60)
    X_train['uid'] = X_train.card1_addr1.astype(str)+'_'+np.floor(X_train.day-X_train.D1).astype(str)

    X_test['day'] = X_test.TransactionDT / (24*60*60)
    X_test['uid'] = X_test.card1_addr1.astype(str)+'_'+np.floor(X_test.day-X_test.D1).astype(str)
    return X_train, X_test

def encode2(X_train, X_test):
    # FREQUENCY ENCODE UID
    encode_FE(X_train,X_test,['uid'])
    # AGGREGATE 
    encode_AG(['TransactionAmt','D4','D9','D10','D15'],['uid'],['mean','std'], X_train, X_test, fillna=True,usena=True)
    # AGGREGATE
    encode_AG(['C'+str(x) for x in range(1,15) if x!=3],['uid'],['mean'],X_train,X_test,fillna=True,usena=True)
    # AGGREGATE
    encode_AG(['M'+str(x) for x in range(1,10)],['uid'],['mean'], X_train, X_test,fillna=True,usena=True)
    # AGGREGATE
    encode_AG2(['P_emaildomain','dist1','DT_M','id_02','cents'], ['uid'], train_df=X_train, test_df=X_test)
    # AGGREGATE
    encode_AG(['C14'],['uid'],['std'],X_train,X_test,fillna=True,usena=True)
    # AGGREGATE 
    encode_AG2(['C13','V314'], ['uid'], train_df=X_train, test_df=X_test)
    # AGGREATE 
    encode_AG2(['V127','V136','V309','V307','V320'], ['uid'], train_df=X_train, test_df=X_test)
    # NEW FEATURE
    X_train['outsider15'] = (np.abs(X_train.D1-X_train.D15)>3).astype('int8')
    X_test['outsider15'] = (np.abs(X_test.D1-X_test.D15)>3).astype('int8')
    print 'outsider15'


    print 'NOW USING THE FOLLOWING',len(cols),'FEATURES.'
    np.array(cols)
    return X_train, X_test

def train96():
    # CHRIS - TRAIN 75% PREDICT 25%
    idxT = X_train.index[:3*len(X_train)//4]
    idxV = X_train.index[3*len(X_train)//4:]

    # KONSTANTIN - TRAIN 4 SKIP 1 PREDICT 1 MONTH
    #idxT = X_train.index[:417559]
    #idxV = X_train.index[-89326:]

def BUILD96_train_and_imp_plot():
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cols)), columns=['Value','Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:50])
    plt.title('XGB96 Most Important')
    plt.tight_layout()
    plt.show()
        
    del clf, h; x=gc.collect()


def BUILD96(X_train, X_test):
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    skf = GroupKFold(n_splits=6)
    for i, (idxT, idxV) in enumerate( skf.split(X_train, y_train, groups=X_train['DT_M']) ):
        month = X_train.iloc[idxV]['DT_M'].iloc[0]
        print 'Fold',i,'withholding month',month
        print ' rows of train =',len(idxT),'rows of holdout =',len(idxV)
        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric='auc',
            # USE CPU
            #nthread=4,
            #tree_method='hist'
            # USE GPU
            tree_method='gpu_hist' 
        )        
        h = clf.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT], 
                eval_set=[(X_train[cols].iloc[idxV],y_train.iloc[idxV])],
                verbose=100, early_stopping_rounds=200)
    
        oof[idxV] += clf.predict_proba(X_train[cols].iloc[idxV])[:,1]
        preds += clf.predict_proba(X_test[cols])[:,1]/skf.n_splits
        del h, clf
        x=gc.collect()
    print '#'*20
    print 'XGB96 OOF CV=',roc_auc_score(y_train,oof)
    return oof, preds


def BUILD96_result_plot():
    plt.hist(oof,bins=100)
    plt.ylim((0,5000))
    plt.title('XGB OOF')
    plt.show()

    X_train['oof'] = oof
    X_train.reset_index(inplace=True)
    X_train[['TransactionID','oof']].to_csv('oof_xgb_96.csv')
    X_train.set_index('TransactionID',drop=True,inplace=True)


def BUILD96_output():
    sample_submission = read_data_csv(r'\samplt_submission', int(sys.argv[1]))
    sample_submission.isFraud = preds
    sample_submission.to_csv(data_path + 'sub_xgb_96.csv',index=False)

    plt.hist(sample_submission.isFraud,bins=100)
    plt.ylim((0,5000))
    plt.title('XGB96 Submission')
    # plt.show()

def test():
    X_test['isFraud'] = sample_submission.isFraud.values
    X_train['isFraud'] = y_train.values
    comb = pd.concat([X_train[['isFraud']],X_test[['isFraud']]],axis=0)

    uids = pd.read_csv('/kaggle/input/ieee-submissions-and-uids/uids_v4_no_multiuid_cleaning..csv',usecols=['TransactionID','uid']).rename({'uid':'uid2'},axis=1)
    comb = comb.merge(uids,on='TransactionID',how='left')
    mp = comb.groupby('uid2').isFraud.agg(['mean'])
    comb.loc[comb.uid2>0,'isFraud'] = comb.loc[comb.uid2>0].uid2.map(mp['mean'])

    uids = pd.read_csv('/kaggle/input/ieee-submissions-and-uids/uids_v1_no_multiuid_cleaning.csv',usecols=['TransactionID','uid']).rename({'uid':'uid3'},axis=1)
    comb = comb.merge(uids,on='TransactionID',how='left')
    mp = comb.groupby('uid3').isFraud.agg(['mean'])
    comb.loc[comb.uid3>0,'isFraud'] = comb.loc[comb.uid3>0].uid3.map(mp['mean'])

    sample_submission.isFraud = comb.iloc[len(X_train):].isFraud.values
    sample_submission.to_csv('sub_xgb_96_PP.csv',index=False)

def main():
    df_train, df_test = read_all_data(int(sys.argv[1]))
    df_train, df_test = data_normalize(df_train, df_test)
    print df_train.shape, df_test.shape
    df_train, df_test = encode(df_train, df_test)
    cols = remove_cols(df_train)
    df_train, df_test = get_uid(df_train, df_test)
    df_train, df_test = encode2(df_train, df_test)
    oof, preds = BUILD96(df_train, df_test)

main()

