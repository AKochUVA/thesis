# Package Imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


def load_nguyen(filename: str):
    # Import data csvs
    path = '../data/nguyen/' + filename
    formula = int(re.search(r'nguyen(\d+)', filename).group(1))
    if formula <= 8:
        return pd.read_csv(path, header=None, names=['x', 'result'])
    elif formula <= 12:
        return pd.read_csv(path, header=None, names=['x', 'y', 'result'])


def nguyen_formula(x, y):
    # depending on nguyen benchmark, select correct formula
    formula = int(re.search(r'nguyen(\d+)', filename).group(1))
    if formula == 1:
        return x ** 3 + x ** 2 + x
    elif formula == 2:
        return x ** 4 + x ** 3 + x ** 2 + x
    elif formula == 7:
        return np.log(x+1) + np.log(x**2 + 1)
    elif formula == 9:
        return np.sin(x) + np.sin(y**2)
    elif formula == 10:
        return 2 * np.sin(x) * np.cos(y)
    else:
        raise ValueError('Formula not defined.')


def print_distribution(x: pd.Series, y: pd.Series = None):
    # Returns the distribution of Series x and y
    print(f'x is distributed between {round(x.min(),2)} and {round(x.max(),2)}, '
          f'with mean {round(x.mean(),2)} and standard deviation {round(x.std(),2)}')
    if y is not None:
        print(f'y is distributed between {round(y.min(), 2)} and {round(y.max(), 2)}, '
              f'with mean {round(y.mean(), 2)} and standard deviation {round(y.std(), 2)}')

def create_nguyen(path: str):
    # Nguyen F11
    x = np.random.uniform(low=0, high=10, size=5000)
    y = np.random.uniform(low=-5, high=5, size=5000)
    result = x ** y

    # Nguyen F12
    # x = np.random.uniform(low=-20, high=20, size=5000)
    # y = np.random.uniform(low=-20, high=20, size=5000)
    # result = np.array(x**4 - x**3 + y**2 / 2 - y)

    df = pd.DataFrame({'x': x, 'y': y, 'result': result})
    df.to_csv(path_or_buf=path, header=False, index=False)


def create_logistic_regression(path, threshold=0.5):
    """Function to create a data set to evaluate the binary classification model"""
    x = np.random.uniform(low=-5, high=5, size=5000)
    result = 1 / (1 + np.exp(-x+2))
    df = pd.DataFrame({'x': x, 'result': result})
    #df['binary_result'] = (df['result'] > threshold).astype(int)
    #df = df.drop(columns=['result'])
    df.to_csv(path_or_buf=path, header=False, index=False)



if __name__ == '__main__':
    path = '../data/logistic/log_reg2_test.csv'
    create_logistic_regression(path)
