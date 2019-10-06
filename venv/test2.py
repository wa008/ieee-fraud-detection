#coding=utf-8
import numpy as np
import pandas as pd
import math
import random
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# import xgboost
# machine learning
# import xgboost
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")
data_path = r'D:\kaggle\data\ieee-fraud-detection'
x = np.arange(1,0,-0.001)
y = (-3 * x * np.log(x) + np.exp(-(40 * (x - 1 / np.e)) ** 4) / 25) / 2 #注意这里在1/e取极值，给它一个智力的波动
plt.figure(figsize=(5,7))
plt.plot(y,x,'r-',linewidth = 2) #注意这里是y，x
plt.grid(True)
plt.title('胸型线',fontsize = 20)
plt.show()
