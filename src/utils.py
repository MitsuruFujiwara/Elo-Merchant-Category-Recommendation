
import os
import pandas as pd
import numpy as np
import requests
import pickle

from sklearn.metrics import mean_squared_error
from time import time, sleep

NUM_FOLDS = 5

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id']

COMPETITION_NAME = 'elo-merchant-category-recommendation'

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# correlationの高い変数を削除する機能
def removeCorrelatedVariables(data, threshold):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    col_drop = [column for column in upper.columns if any(upper[column] > threshold) & ('visitors' not in column)]
    return col_drop

# 欠損値の率が高い変数を削除する機能
def removeMissingVariables(data, threshold):
    missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    col_missing = missing.index[missing > 0.75]
    col_missing = [column for column in col_missing if 'visitors' not in column]

# LINE通知用
def line_notify(message):
    f = open('../input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    print(message)

# API経由でsubmitする機能 https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/py/utils.py
def submit(file_path, comment='from API'):
    os.system('kaggle competitions submit -c {} -f {} -m "{}"'.format(COMPETITION_NAME,file_path,comment))
    sleep(60) # tekito~~~~
    tmp = os.popen('kaggle competitions submissions -c {} -v | head -n 2'.format(COMPETITION_NAME)).read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += '{}: {}\n'.format(i,j)
#        print(f'{i}: {j}') # TODO: comment out later?
    line_notify(message.rstrip())

# save pkl
def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close

# load pkl
def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out
