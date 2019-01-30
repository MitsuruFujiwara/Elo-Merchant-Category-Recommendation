import datetime
import gc
import numpy as np
import os
import pandas as pd
import time

from workalendar.america import Brazil

from features.base import Feature, get_arguments, generate_features
from utils import one_hot_encoder, reduce_mem_usage

# convert csv to feather
def convert_to_feather(filepaths):
    for path in filepaths:
        if os.path.exists(path):
            continue
        else:
            (pd.read_csv(path)).to_feather()

# train & test
class TrainTestBasic(Feature):
    def create_features(self):
        # load csv
        # train_df = pd.read_csv('../input/train.csv', index_col=['card_id'], nrows=num_rows)
        train_df = feather.read_dataframe('../data/input/train.feather')
        train_df = train_df.set_index('card_id')
        # test_df = pd.read_csv('../input/test.csv', index_col=['card_id'], nrows=num_rows)
        test_df = feather.read_dataframe('../data/input/test.feather')
        test_df = test_df.set_index('card_id')

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
    #    df['month'] = df['first_active_month'].dt.month.fillna(0).astype(int).astype(object)
    #    df['year'] = df['first_active_month'].dt.year.fillna(0).astype(int).astype(object)
    #    df['dayofweek'] = df['first_active_month'].dt.dayofweek.fillna(0).astype(int).astype(object)
    #    df['weekofyear'] = df['first_active_month'].dt.weekofyear.fillna(0).astype(int).astype(object)
        df['quarter'] = df['first_active_month'].dt.quarter
    #    df['month_year'] = df['month'].astype(str)+'_'+df['year'].astype(str)
        df['elapsed_time'] = (pd.to_datetime('2019-01-18') - df['first_active_month']).dt.days

        df['days_feature1'] = df['elapsed_time'] * df['feature_1']
        df['days_feature2'] = df['elapsed_time'] * df['feature_2']
        df['days_feature3'] = df['elapsed_time'] * df['feature_3']

        df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
        df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
        df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

        # one hot encoding
        df, cols = one_hot_encoder(df, nan_as_category=False)

        for f in ['feature_1','feature_2','feature_3']:
            order_label = df.groupby([f])['outliers'].mean()
            df[f] = df[f].map(order_label)

        df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
        df['feature_mean'] = df['feature_sum']/3
        df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
        df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
        df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

        # df = df.reset_index()
        # self.df = df
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        self.train = train_df.reset_index()
        self.test = test_df.reset_index()

if __name__ == '__main__':
    generate_features(globals())
