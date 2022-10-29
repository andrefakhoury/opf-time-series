import numpy as np
import pandas as pd
from dtaidistance import dtw

def error(Y, Y_pred):
    return (len(Y) - np.count_nonzero(Y == Y_pred))/len(Y)

# 1-NN classify
def euclidean_distance_classify(X, y, X_test):
    preds = []
    for x_test in X_test:
        best_idx, best_dist = -1, np.inf
        for idx, x in enumerate(X):
            cur_dist = np.linalg.norm(x - x_test)
            if best_idx == -1 or cur_dist < best_dist:
                best_idx, best_dist = idx, cur_dist
        preds.append(y[best_idx])
    return preds


# 1-NN classify. DTW w=10%
def dtw_distance_classify(X, y, X_test):
    preds = []
    for x_test in X_test:
        best_idx, best_dist = -1, np.inf
        for idx, x in enumerate(X):
            cur_dist = dtw.distance_fast(x, x_test, window=int(0.1*len(x)), use_pruning=True)
            if best_idx == -1 or cur_dist < best_dist:
                best_idx, best_dist = idx, cur_dist
        preds.append(y[best_idx])
    return preds

def read_df(df_name):
    df = pd.read_table(f'data/UCRArchive_2018/{df_name}/{df_name}_TRAIN.tsv', header=None)
    df_test = pd.read_table(f'data/UCRArchive_2018/{df_name}/{df_name}_TEST.tsv', header=None)
    datasets_df = pd.read_csv('data/DataSummary.csv')

    X, _y = df.iloc[:, 1:], df.iloc[:, 0]
    X_test, _y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]
    
    # replace nan values with mean of row in both X and X_test
    X = X.where(X.notna(), X.mean(axis=1), axis=0)
    X_test = X_test.where(X_test.notna(), X_test.mean(axis=1), axis=0)

    X = np.array(X, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y = np.array(_y, dtype=int)
    y_test = np.array(_y_test, dtype=int)

    all_y = np.unique(np.append(y, y_test))
    mp_y = {all_y[i]: i+1 for i in range(len(all_y))}

    for k, v in mp_y.items():
        y[_y == k] = v
        y_test[_y_test == k] = v
        
    df_row = datasets_df.loc[datasets_df['Name'] == df_name]
    df_errors = {
        "ED": df_row.iloc[:, 7].values[0],
        "DTW": df_row.iloc[:, 8].values[0],
        "DTW_W100": df_row.iloc[:, 9].values[0],
    }

    return X, y, X_test, y_test, df_errors