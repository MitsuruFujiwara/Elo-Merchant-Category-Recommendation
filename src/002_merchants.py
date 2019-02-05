
import datetime
import feather
import gc
import pandas as pd
import numpy as np
import warnings

from utils import one_hot_encoder, save2pkl

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(num_rows=None):
    # load csv
    merchants_df = pd.read_csv('../input/merchants.csv',index_col=['merchant_id'],nrows=num_rows)

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

    aggs['numerical_1'] = ['sum','max','min','mean','var','skew']
    aggs['numerical_2'] = ['sum','max','min','mean','var','skew']
    aggs['avg_sales_lag3'] = ['sum','max','min','mean','var','skew']
    aggs['avg_sales_lag6'] = ['sum','max','min','mean','var','skew']
    aggs['avg_sales_lag12'] = ['sum','max','min','mean','var','skew']
    aggs['avg_purchases_lag3'] = ['sum','max','min','mean','var','skew']
    aggs['avg_purchases_lag6'] = ['sum','max','min','mean','var','skew']
    aggs['avg_purchases_lag12'] = ['sum','max','min','mean','var','skew']
    aggs['active_months_lag3'] = ['sum','max','min','mean','var','skew']
    aggs['active_months_lag6'] = ['sum','max','min','mean','var','skew']
    aggs['active_months_lag12'] = ['sum','max','min','mean','var','skew']
    aggs['category_1'] = ['sum','mean']
    aggs['category_4'] = ['sum','mean']
    """
    aggs['most_recent_sales_range_A'] = ['sum','mean']
    aggs['most_recent_sales_range_B'] = ['sum','mean']
    aggs['most_recent_sales_range_C'] = ['sum','mean']
    aggs['most_recent_sales_range_D'] = ['sum','mean']
    aggs['most_recent_sales_range_E'] = ['sum','mean']
    aggs['most_recent_purchases_range_A'] = ['sum','mean']
    aggs['most_recent_purchases_range_B'] = ['sum','mean']
    aggs['most_recent_purchases_range_C'] = ['sum','mean']
    aggs['most_recent_purchases_range_D'] = ['sum','mean']
    aggs['most_recent_purchases_range_E'] = ['sum','mean']
    """
    aggs['category_2_-1'] = ['sum','mean']
    aggs['category_2_1'] = ['sum','mean']
    aggs['category_2_2'] = ['sum','mean']
    aggs['category_2_3'] = ['sum','mean']
    aggs['category_2_4'] = ['sum','mean']
    aggs['category_2_5'] = ['sum','mean']
    aggs['avg_numerical'] = ['sum','max','min','mean','var','skew']
    aggs['avg_sales'] = ['sum','max','min','mean','var','skew']
    aggs['avg_purchases'] = ['sum','max','min','mean','var','skew']
    aggs['avg_active_months'] = ['sum','max','min','mean','var','skew']
    aggs['max_sales'] = ['sum','max','min','mean','var','skew']
    aggs['max_purchases'] = ['sum','max','min','mean','var','skew']
    aggs['max_active_months'] = ['sum','max','min','mean','var','skew']
    aggs['min_sales'] = ['sum','max','min','mean','var','skew']
    aggs['min_purchases'] = ['sum','max','min','mean','var','skew']
    aggs['min_active_months'] = ['sum','max','min','mean','var','skew']
    aggs['sum_category'] = ['sum','mean']

    merchants_df = merchants_df.reset_index().groupby('merchant_id').agg(aggs)

    # カラム名の変更
    merchants_df.columns = pd.Index([e[0] + "_" + e[1] for e in merchants_df.columns.tolist()])
    merchants_df.columns = ['mer_'+ c for c in merchants_df.columns]

    # save
    save2pkl('../features/merchants.pkl', merchants_df)

if __name__ == '__main__':
    main()
