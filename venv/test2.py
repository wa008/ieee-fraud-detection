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


x = range(10)
y = range(1, 20, 2)
print x, y
plt.scatter(x, y)
plt.show()