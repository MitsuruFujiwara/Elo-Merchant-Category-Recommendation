
import pandas as pd
import numpy as np
import gc
import os

from utils import one_hot_encoder

################################################################################
# 提供データを読み込み、データに前処理を施し、モデルに入力が可能な状態でファイル出力するモジュール。
# get_train_dataやget_test_dataのように、学習用と評価用を分けて、前処理を行う関数を定義してください。
################################################################################

# preprocessing train & test
def train_test(num_rows=None):

    # load csv
    train_df = pd.read_csv('../input/train.csv', index_col=['card_id'], nrows=num_rows)
    test_df = pd.read_csv('../input/test.csv', index_col=['card_id'], nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # set target as nan
    test_df['target'] = np.nan

    # merge
    df = train_df.append(test_df)

    del train_df, test_df
    gc.collect()

    # datetimeへ変換
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    # datetime features
    df['month'] = df['first_active_month'].dt.month.fillna(0).astype(int).astype(object)
    df['year'] = df['first_active_month'].dt.year.fillna(0).astype(int).astype(object)
    df['month_year'] = df['month'].astype(str)+'_'+df['year'].astype(str)

    # features1~3
    df['feature_1'] = df['feature_1'].astype(object)
    df['feature_2'] = df['feature_2'].astype(object)

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    return df

# preprocessing historical transactions
def historical_transactions(num_rows=None):
    # load csv
    hist_df = pd.read_csv('../input/historical_transactions.csv', nrows=num_rows)

    # TODO:

    return hist_df

# preprocessing merchants
def merchants(num_rows=None):
    # load csv
    merchants_df = pd.read_csv('../input/merchants.csv', nrows=num_rows)

    # TODO:

    return merchants_df

# preprocessing new_merchant_transactions
def new_merchant_transactions(num_rows=None):
    # load csv
    new_merchant_df = pd.read_csv('../input/new_merchant_transactions.csv', nrows=num_rows)

    # TODO:

    return new_merchant_df

if __name__ == '__main__':
    # test
    num_rows=10000

    # train & test
    df = train_test(num_rows)

    print(df)
