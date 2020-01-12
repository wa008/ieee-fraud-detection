# ieee notebook(from first place)

1. feature engineering
   1. 特征编码
      1. agg(min, max, mean, std)
      2. frequence encode
      3. label encode
      4. combine encode
   2. correlation analysis
      1. heat map（correlation analysis）
      2. 根据NaN的数量，判断特征之间的相似性（楼主是将缺失值数量相同的特征作为相似特征
   3. feature selection
      1. 对每一个特征单独进行训练，以训练集第一个月的数据作为训练集，最后一个月数据作为验证集，选取训练集验证集上auc>0.5的特征，以此会剔除19个特征。
      2. forward feature selection：从有到无添加特征，知道添加新的特征也不能提升模型的效果
      3. Recursive Feature Elimination（递归特征消除）：Recursive feature elimination (RFE) is a feature selection method that fits a model and removes the weakest feature (or features) until the specified number of features is reached. Features are ranked by the model’s `coef_` or `feature_importances_` attributes, and by recursively eliminating a small number of features per loop, RFE attempts to eliminate dependencies and collinearity that may exist in the model.（从多到少，减少特征数量，直到达到指定的特征数量，RFE会尝试消除模型中可能存在的依赖性和共线性）
      4. permutation importance（排列重要性）：去掉某个特征，查看模型在训练测试集上表现的变化
         1. 可用于防止模型过拟合
         2. 在对效率要求高的场景下，可以尝试减少特征数量
      5. adversarial validation（对抗性验证）：选择与测试集最相似的数据作为验证集，★★★
      6. correlation analysis：heat map
      7. time consistency（时间一致性）："time consistency" is to train a single model using a single feature (or small group of features) on the first month of train dataset and predict `isFraud` for the last month of train dataset, This evaluates whether a feature by itself is consistent over time（这个评估一个特征是否随时间不变）
      8. client consistency：构造client id
      9. train/test distribution analysis
2. trick
   1. 构造client id作为唯一标识，绝大多情况下同一个uid下的交易，要么都作弊，要么都不作弊，对同一个client id下的其他特征进行【特征编码】
   2. Validation Strategy
      1. Train on first x months of train, skip x month, predict x month
      2. 看各个模型在know, unknow and Questionable client id上的表现，
         1. 如何定义Questionable client id？？？
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

# 特征权重可视化
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cols)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:50])
plt.title('XGB95 Most Important Features')
plt.tight_layout()
plt.show()
del clf, h; x=gc.collect()
```

4. 可视化
   1. 多个子图可视化
   2. heat map