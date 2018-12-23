
import pandas as pd
import numpy as np
import gc
import os
import time

from contextlib import contextmanager
from utils import one_hot_encoder
import datetime

################################################################################
# 提供データを読み込み、データに前処理を施し、モデルに入力が可能な状態でファイル出力するモジュール。
# get_train_dataやget_test_dataのように、学習用と評価用を分けて、前処理を行う関数を定義してください。
################################################################################

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

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
    hist_df = pd.read_csv('../input/historical_transactions.csv', nrows=num_rows, index_col=['card_id'])

    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('A',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

    # Y/Nのカラムを1-0へ変換
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['year'] = hist_df['purchase_date'].dt.year
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']

    col_unique =['month', 'hour', 'weekofyear', 'weekday', 'year', 'day',
                 'subsector_id', 'merchant_id', 'merchant_category_id']
    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    aggs['purchase_amount'] = ['sum','max','min','mean','var']
    aggs['installments'] = ['sum','max','min','mean','var']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var']
    aggs['month_diff'] = ['mean']
    aggs['authorized_flag'] = ['sum', 'mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']

    for col in ['category_2','category_3']:
        hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        aggs[col+'_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)


    # カラム名の変更
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_'+ c for c in hist_df.columns]

    hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
    hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
    hist_df['hist_purchase_date_uptonow'] = (datetime.datetime.today()-hist_df['hist_purchase_date_max']).dt.days

    return hist_df

# preprocessing merchants
def merchants(num_rows=None):
    # load csv
    merchants_df = pd.read_csv('../input/merchants.csv', nrows=num_rows)

    # fillna
    merchants_df['category_2'].fillna(1.0,inplace=True)
    merchants_df['category_3'].fillna('A',inplace=True)
    merchants_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

    # Y/Nのカラムを1-0へ変換
    merchants_df['authorized_flag'] = merchants_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    merchants_df['category_1'] = merchants_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)

    # datetime features
    merchants_df['purchase_date'] = pd.to_datetime(merchants_df['purchase_date'])

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
    with timer("train & test"):
        df = train_test(num_rows)
    with timer("historical transactions"):
        df = pd.merge(df, historical_transactions(num_rows), on='card_id', how='outer')
#    with timer("merchants"):
#        df = pd.merge(df, merchants(num_rows), on='card_id', how='outer')
#    with timer("merchants"):
#        df = pd.merge(df, new_merchant_transactions(num_rows), on='card_id', how='outer')

    # train & test
    df = train_test(num_rows)

    print(df)
