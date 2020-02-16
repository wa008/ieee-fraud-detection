import pandas as pd
import numpy as np

def change_data(x):
    x['c'] = x['a'] + 1

def main():
    df = pd.DataFrame(data = np.arange(6).reshape(3, 2), columns = ['a', 'b'])
    print np.arange(6)
    print df
    change_data(df)
    print df

main()