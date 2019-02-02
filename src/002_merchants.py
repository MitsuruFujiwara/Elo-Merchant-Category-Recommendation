
import datetime
import feather
import gc
import pandas as pd
import numpy as np
import warnings

from utils import one_hot_encoder

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # load csv
    merchants_df = pd.read_csv('../input/merchants.csv',index_col=['merchant_id'])

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

    # one hot encoding
    merchants_df, cols = one_hot_encoder(merchants_df, nan_as_category=False)

    # unique columns
    col_unique =['merchant_group_id', 'merchant_category_id', 'subsector_id',
                 'city_id', 'state_id']

    # aggregation
    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    aggs['numerical_1'] = ['mean','max','min','std','var']
    aggs['numerical_2'] = ['mean','max','min','std','var']
    aggs['avg_sales_lag3'] = ['mean','max','min','std','var']
    aggs['avg_sales_lag6'] = ['mean','max','min','std','var']
    aggs['avg_sales_lag12'] = ['mean','max','min','std','var']
    aggs['avg_purchases_lag3'] = ['mean','max','min','std','var']
    aggs['avg_purchases_lag6'] = ['mean','max','min','std','var']
    aggs['avg_purchases_lag12'] = ['mean','max','min','std','var']
    aggs['active_months_lag3'] = ['mean','max','min','std','var']
    aggs['active_months_lag6'] = ['mean','max','min','std','var']
    aggs['active_months_lag12'] = ['mean','max','min','std','var']
    aggs['category_1'] = ['mean']
    aggs['category_4'] = ['mean']
    aggs['most_recent_sales_range_A'] = ['mean']
    aggs['most_recent_sales_range_B'] = ['mean']
    aggs['most_recent_sales_range_C'] = ['mean']
    aggs['most_recent_sales_range_D'] = ['mean']
    aggs['most_recent_sales_range_E'] = ['mean']
    aggs['most_recent_purchases_range_A'] = ['mean']
    aggs['most_recent_purchases_range_B'] = ['mean']
    aggs['most_recent_purchases_range_C'] = ['mean']
    aggs['most_recent_purchases_range_D'] = ['mean']
    aggs['most_recent_purchases_range_E'] = ['mean']
    aggs['category_2_-1'] = ['mean']
    aggs['category_2_1'] = ['mean']
    aggs['category_2_2'] = ['mean']
    aggs['category_2_3'] = ['mean']
    aggs['category_2_4'] = ['mean']
    aggs['category_2_5'] = ['mean']
    aggs['avg_numerical'] = ['mean','max','min','std','var']
    aggs['avg_sales'] = ['mean','max','min','std','var']
    aggs['avg_purchases'] = ['mean','max','min','std','var']
    aggs['avg_active_months'] = ['mean','max','min','std','var']
    aggs['max_sales'] = ['mean','max','min','std','var']
    aggs['max_purchases'] = ['mean','max','min','std','var']
    aggs['max_active_months'] = ['mean','max','min','std','var']
    aggs['min_sales'] = ['mean','max','min','std','var']
    aggs['min_purchases'] = ['mean','max','min','std','var']
    aggs['min_active_months'] = ['mean','max','min','std','var']
    aggs['sum_category'] = ['mean']

    merchants_df = merchants_df.reset_index().groupby('merchant_id').agg(aggs)

    # カラム名の変更
    merchants_df.columns = pd.Index([e[0] + "_" + e[1] for e in merchants_df.columns.tolist()])
    merchants_df.columns = ['mer_'+ c for c in merchants_df.columns]

    merchants_df.reset_index(inplace=True)

    # save
    merchants_df.to_feather('../features/merchants.feather')

if __name__ == '__main__':
    main()
