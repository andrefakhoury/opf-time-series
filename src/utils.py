import numpy as np
import pandas as pd

def error(Y, Y_pred):
    return (len(Y) - np.count_nonzero(Y == Y_pred))/len(Y)

def read_df(df_name):
    df = pd.read_table(f'data/UCRArchive_2018/{df_name}/{df_name}_TRAIN.tsv', header=None)
    df_test = pd.read_table(f'data/UCRArchive_2018/{df_name}/{df_name}_TEST.tsv', header=None)
    datasets_df = pd.read_csv('data/DataSummary.csv')

    X, _y = df.iloc[:, 1:], df.iloc[:, 0]
    X_test, _y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]
    dataset_error = datasets_df.loc[datasets_df['Name'] == df_name].iloc[:, 7].values[0]

    y = np.array(_y, dtype=int)
    y_test = np.array(_y_test, dtype=int)

    all_y = np.unique(np.append(y, y_test))
    mp_y = {all_y[i]: i+1 for i in range(len(all_y))}

    for k, v in mp_y.items():
        y[_y == k] = v
        y_test[_y_test == k] = v

    return X, y, X_test, y_test, dataset_error