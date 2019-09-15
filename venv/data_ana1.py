#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
from tqdm import tqdm
# machine learning
import xgboost
# myself libs
from kFold_corss import kFold_cross
from write_result import write
# other
import warnings
warnings.filterwarnings("ignore")
from hello1 import read_all_data

def main():
    df_train, df_test = read_all_data('all')
    dt = df_train['TransactionDT'].tolist()
    print np.min(dt)
    print np.max(dt)
    print np.average(dt)

main()