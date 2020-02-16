import pandas as pd
import numpy as np

def change_data(x):
    x['c'] = x['a'] + 1

def main():
    df = pd.DataFrame(data = np.arange(6).reshape(3, 2), columns = ['a', 'b'])
    df['a'] = df['a'].apply(lambda x : 1 if x == 1 else 2 if x == 2 else 3)

    for i in range(10):
        print i, 

main()