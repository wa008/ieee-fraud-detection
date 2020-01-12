# first place notebook

1. feature selection
   1. 对每一个特征单独进行训练，以训练集第一个月的数据作为训练集，最后一个月数据作为验证集，选取训练集验证集上auc>0.5的特征，以此会剔除19个特征
   2. 特征编码方式
      1. agg(min, max, mean, std)
      2. frequence encode
      3. label encode
      4. combine encode
2. trick
   1. 构造client id作为唯一标识，绝大多情况下同一个uid下的交易，要么都作弊，要么都不作弊，对同一个client id下的其他特征进行【特征编码】
3. xgboost
   1. 初始化参数demo+xgb特征权重可视化

```
clf = xgb.XGBClassifier(n_estimators=2000,
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

# 特征权重
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cols)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:50])
plt.title('XGB95 Most Important Features')
plt.tight_layout()
plt.show()
del clf, h; x=gc.collect()
```

