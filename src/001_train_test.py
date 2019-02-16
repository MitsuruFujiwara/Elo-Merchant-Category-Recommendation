
import datetime
import feather
import gc
import pandas as pd
import numpy as np
import warnings

from utils import one_hot_encoder, save2pkl

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # load csv
    train_df = pd.read_csv('../input/train.csv', index_col=['card_id'])
    test_df = pd.read_csv('../input/test.csv', index_col=['card_id'])

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
    df['month_year'] = df['month'].astype(str)+'_'+df['year'].astype(str)
    df['elapsed_time'] = (pd.to_datetime('2018-03-01') - df['first_active_month']).dt.days

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
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    # save
    save2pkl('../features/train_test.pkl', df)

if __name__ == '__main__':
    main()
