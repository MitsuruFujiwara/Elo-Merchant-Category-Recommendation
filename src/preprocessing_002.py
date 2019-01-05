

import datetime
import gc
import numpy as np
import os
import pandas as pd
import time

from contextlib import contextmanager
from workalendar.america import Brazil

from utils import one_hot_encoder, reduce_mem_usage

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

    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30, 'outliers'] = 1

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
    df['dayofweek'] = df['first_active_month'].dt.dayofweek.fillna(0).astype(int).astype(object)
    df['weekofyear'] = df['first_active_month'].dt.weekofyear.fillna(0).astype(int).astype(object)
    df['quarter'] = df['first_active_month'].dt.quarter
    df['is_month_start'] = df['first_active_month'].dt.is_month_start
    df['month_year'] = df['month'].astype(str)+'_'+df['year'].astype(str)

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    for f in ['feature_1','feature_2','feature_3']:
        order_label = df.groupby([f])['outliers'].mean()
        df[f] = df[f].map(order_label)

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum']/3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_std'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    return df

# preprocessing merchants
def merchants(num_rows=None):
    # load csv
    merchants_df = pd.read_csv('../input/merchants.csv', nrows=num_rows, index_col=['merchant_id'])

    # Y/Nのカラムを1-0へ変換
    merchants_df['category_1'] = merchants_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    merchants_df['category_4'] = merchants_df['category_4'].map({'Y': 1, 'N': 0}).astype(int)

    # additional features
    merchants_df['avg_numerical'] = merchants_df[['numerical_1','numerical_2']].mean(axis=1)
    merchants_df['avg_sales'] = merchants_df[['avg_sales_lag3','avg_sales_lag6','avg_sales_lag12']].mean(axis=1)
    merchants_df['avg_purchases'] = merchants_df[['avg_purchases_lag3','avg_purchases_lag6','avg_purchases_lag12']].mean(axis=1)
    merchants_df['avg_active_months'] = merchants_df[['active_months_lag3','active_months_lag6','active_months_lag12']].mean(axis=1)
    merchants_df['max_sales'] = merchants_df[['avg_sales_lag3','avg_sales_lag6','avg_sales_lag12']].max(axis=1)
    merchants_df['max_purchases'] = merchants_df[['avg_purchases_lag3','avg_purchases_lag6','avg_purchases_lag12']].max(axis=1)
    merchants_df['max_active_months'] = merchants_df[['active_months_lag3','active_months_lag6','active_months_lag12']].max(axis=1)
    merchants_df['min_sales'] = merchants_df[['avg_sales_lag3','avg_sales_lag6','avg_sales_lag12']].min(axis=1)
    merchants_df['min_purchases'] = merchants_df[['avg_purchases_lag3','avg_purchases_lag6','avg_purchases_lag12']].min(axis=1)
    merchants_df['min_active_months'] = merchants_df[['active_months_lag3','active_months_lag6','active_months_lag12']].min(axis=1)
    merchants_df['sum_category'] = merchants_df[['category_1','category_2','category_4']].sum(axis=1)

    # fillna
    merchants_df['category_2'] = merchants_df['category_2'].fillna(-1).astype(int).astype(object)

    # memory usage削減
    merchants_df = reduce_mem_usage(merchants_df)

    # one hot encoding
    merchants_df, cols = one_hot_encoder(merchants_df, nan_as_category=False)

    # unique columns
    col_unique =['merchant_group_id', 'merchant_category_id', 'subsector_id',
                 'city_id', 'state_id']

    # aggregation
    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    aggs['numerical_1'] = ['mean','sum','max','min','std']
    aggs['numerical_2'] = ['mean','sum','max','min','std']
    aggs['avg_sales_lag3'] = ['mean','sum','max','min','std']
    aggs['avg_sales_lag6'] = ['mean','sum','max','min','std']
    aggs['avg_sales_lag12'] = ['mean','sum','max','min','std']
    aggs['avg_purchases_lag3'] = ['mean','sum','max','min','std']
    aggs['avg_purchases_lag6'] = ['mean','sum','max','min','std']
    aggs['avg_purchases_lag12'] = ['mean','sum','max','min','std']
    aggs['active_months_lag3'] = ['mean','sum','max','min','std']
    aggs['active_months_lag6'] = ['mean','sum','max','min','std']
    aggs['active_months_lag12'] = ['mean','sum','max','min','std']
    aggs['category_1'] = ['mean','sum']
    aggs['category_4'] = ['mean','sum']
    aggs['most_recent_sales_range_A'] = ['mean','sum']
    aggs['most_recent_sales_range_B'] = ['mean','sum']
    aggs['most_recent_sales_range_C'] = ['mean','sum']
    aggs['most_recent_sales_range_D'] = ['mean','sum']
    aggs['most_recent_sales_range_E'] = ['mean','sum']
    aggs['most_recent_purchases_range_A'] = ['mean','sum']
    aggs['most_recent_purchases_range_B'] = ['mean','sum']
    aggs['most_recent_purchases_range_C'] = ['mean','sum']
    aggs['most_recent_purchases_range_D'] = ['mean','sum']
    aggs['most_recent_purchases_range_E'] = ['mean','sum']
    aggs['category_2_-1'] = ['mean','sum']
    aggs['category_2_1'] = ['mean','sum']
    aggs['category_2_2'] = ['mean','sum']
    aggs['category_2_3'] = ['mean','sum']
    aggs['category_2_4'] = ['mean','sum']
    aggs['category_2_5'] = ['mean','sum']
    aggs['avg_numerical'] = ['mean','sum','max','min','std']
    aggs['avg_sales'] = ['mean','sum','max','min','std']
    aggs['avg_purchases'] = ['mean','sum','max','min','std']
    aggs['avg_active_months'] = ['mean','sum','max','min','std']
    aggs['max_sales'] = ['mean','sum','max','min','std']
    aggs['max_purchases'] = ['mean','sum','max','min','std']
    aggs['max_active_months'] = ['mean','sum','max','min','std']
    aggs['min_sales'] = ['mean','sum','max','min','std']
    aggs['min_purchases'] = ['mean','sum','max','min','std']
    aggs['min_active_months'] = ['mean','sum','max','min','std']
    aggs['sum_category'] = ['mean','sum']

    merchants_df = merchants_df.reset_index().groupby('merchant_id').agg(aggs)

    # カラム名の変更
    merchants_df.columns = pd.Index([e[0] + "_" + e[1] for e in merchants_df.columns.tolist()])
    merchants_df.columns = ['mer_'+ c for c in merchants_df.columns]

    # memory usage削減
    merchants_df = reduce_mem_usage(merchants_df)

    return merchants_df

# preprocessing historical transactions
def historical_transactions(merchants_df, num_rows=None):
    # load csv
    hist_df = pd.read_csv('../input/historical_transactions.csv', nrows=num_rows)

    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('A',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # Y/Nのカラムを1-0へ変換
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A':0, 'B':1, 'C':2})

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['year'] = hist_df['purchase_date'].dt.year
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

    #ブラジルの休日
    cal = Brazil()
    hist_df['is_holiday'] = hist_df['purchase_date'].dt.date.apply(cal.is_holiday).astype(int)

    # 購入日からイベント日までの経過日数
    #Christmas : December 25 2017
    hist_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 14 2017
    hist_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #fathers day: August 13 2017
    hist_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    hist_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Valentine's Day : 12th June, 2017
    hist_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    hist_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    #2018
    #Mothers Day: May 13 2018
    hist_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']

    # merge merchants_df
    hist_df = pd.merge(hist_df, merchants_df, on='merchant_id', how='outer')
    merchants_cols = merchants_df.columns.tolist()
    del merchants_df
    gc.collect()

    # memory usage削減
    hist_df = reduce_mem_usage(hist_df)

    col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'year', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    for col in merchants_cols:
        aggs[col] = ['sum', 'mean']

    aggs['purchase_amount'] = ['sum','max','min','mean','var', 'std', 'skew']
    aggs['installments'] = ['sum','max','min','mean','var', 'std', 'skew']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var', 'std', 'skew']
    aggs['month_diff'] = ['max','min','mean','var', 'std', 'skew']
    aggs['authorized_flag'] = ['sum', 'mean', 'min', 'max']
    aggs['weekend'] = ['sum', 'mean', 'min', 'max']
    aggs['category_1'] = ['sum', 'mean', 'min']
    aggs['category_2'] = ['sum', 'mean', 'min']
    aggs['category_3'] = ['sum', 'mean', 'min']
    aggs['card_id'] = ['size','count']
    aggs['is_holiday'] = ['sum', 'mean']
    aggs['price'] = ['sum','mean','max','min','std']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']

    for col in ['category_2','category_3']:
        hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # カラム名の変更
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_'+ c for c in hist_df.columns]

    hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
    hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
    hist_df['hist_purchase_date_uptonow'] = (datetime.datetime.today()-hist_df['hist_purchase_date_max']).dt.days
    hist_df['hist_purchase_date_uptomin'] = (datetime.datetime.today()-hist_df['hist_purchase_date_min']).dt.days

    # memory usage削減
    hist_df = reduce_mem_usage(hist_df)

    return hist_df

# preprocessing new_merchant_transactions
def new_merchant_transactions(merchants_df, num_rows=None):
    # load csv
    new_merchant_df = pd.read_csv('../input/new_merchant_transactions.csv', nrows=num_rows)

    # fillna
    new_merchant_df['category_2'].fillna(1.0,inplace=True)
    new_merchant_df['category_3'].fillna('A',inplace=True)
    new_merchant_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    new_merchant_df['installments'].replace(-1, np.nan,inplace=True)
    new_merchant_df['installments'].replace(999, np.nan,inplace=True)

    # Y/Nのカラムを1-0へ変換
    new_merchant_df['authorized_flag'] = new_merchant_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_1'] = new_merchant_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_3'] = new_merchant_df['category_3'].map({'A':0, 'B':1, 'C':2}).astype(int)

    # datetime features
    new_merchant_df['purchase_date'] = pd.to_datetime(new_merchant_df['purchase_date'])
    new_merchant_df['year'] = new_merchant_df['purchase_date'].dt.year
    new_merchant_df['month'] = new_merchant_df['purchase_date'].dt.month
    new_merchant_df['day'] = new_merchant_df['purchase_date'].dt.day
    new_merchant_df['hour'] = new_merchant_df['purchase_date'].dt.hour
    new_merchant_df['weekofyear'] = new_merchant_df['purchase_date'].dt.weekofyear
    new_merchant_df['weekday'] = new_merchant_df['purchase_date'].dt.weekday
    new_merchant_df['weekend'] = (new_merchant_df['purchase_date'].dt.weekday >=5).astype(int)

    # additional features
    new_merchant_df['price'] = new_merchant_df['purchase_amount'] / new_merchant_df['installments']

    #ブラジルの休日
    cal = Brazil()
    new_merchant_df['is_holiday'] = new_merchant_df['purchase_date'].dt.date.apply(cal.is_holiday).astype(int)

    # 購入日からイベント日までの経過日数
    #Christmas : December 25 2017
    new_merchant_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 14 2017
    new_merchant_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #fathers day: August 13 2017
    new_merchant_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    new_merchant_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Valentine's Day : 12th June, 2017
    new_merchant_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    new_merchant_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    #2018
    #Mothers Day: May 13 2018
    new_merchant_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    new_merchant_df['month_diff'] = ((datetime.datetime.today() - new_merchant_df['purchase_date']).dt.days)//30
    new_merchant_df['month_diff'] += new_merchant_df['month_lag']

    # merge merchants df
    merchants_cols = merchants_df.columns.tolist()
    new_merchant_df = pd.merge(new_merchant_df, merchants_df, on='merchant_id', how='outer')
    del merchants_df
    gc.collect()

    # memory usage削減
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'year', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    for col in merchants_cols:
        aggs[col] = ['sum', 'mean']

    aggs['purchase_amount'] = ['sum','max','min','mean','var','std','skew']
    aggs['installments'] = ['sum','max','min','mean','var','std','skew']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var','std','skew']
    aggs['month_diff'] = ['max','min','mean','var','std','skew']
    aggs['authorized_flag'] = ['sum', 'mean', 'min', 'max']
    aggs['weekend'] = ['sum', 'mean', 'min', 'max']
    aggs['category_1'] = ['sum', 'mean', 'min']
    aggs['category_2'] = ['sum', 'mean', 'min']
    aggs['category_3'] = ['sum', 'mean', 'min']
    aggs['card_id'] = ['size','count']
    aggs['is_holiday'] = ['sum', 'mean']
    aggs['price'] = ['sum','mean','max','min','std']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']

    for col in ['category_2','category_3']:
        new_merchant_df[col+'_mean'] = new_merchant_df.groupby([col])['purchase_amount'].transform('mean')
        new_merchant_df[col+'_min'] = new_merchant_df.groupby([col])['purchase_amount'].transform('min')
        new_merchant_df[col+'_max'] = new_merchant_df.groupby([col])['purchase_amount'].transform('max')
        new_merchant_df[col+'_sum'] = new_merchant_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    new_merchant_df = new_merchant_df.reset_index().groupby('card_id').agg(aggs)

    # カラム名の変更
    new_merchant_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_merchant_df.columns.tolist()])
    new_merchant_df.columns = ['new_'+ c for c in new_merchant_df.columns]

    new_merchant_df['new_purchase_date_diff'] = (new_merchant_df['new_purchase_date_max']-new_merchant_df['new_purchase_date_min']).dt.days
    new_merchant_df['new_purchase_date_average'] = new_merchant_df['new_purchase_date_diff']/new_merchant_df['new_card_id_size']
    new_merchant_df['new_purchase_date_uptonow'] = (datetime.datetime.today()-new_merchant_df['new_purchase_date_max']).dt.days
    new_merchant_df['new_purchase_date_uptomin'] = (datetime.datetime.today()-new_merchant_df['new_purchase_date_min']).dt.days

    # memory usage削減
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    return new_merchant_df

# additional features
def additional_features(df):
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    date_features=['hist_purchase_date_max','hist_purchase_date_min',
                   'new_purchase_date_max', 'new_purchase_date_min']

    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size']+df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count']+df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max']+df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min']+df['hist_purchase_amount_min']

    return df

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
