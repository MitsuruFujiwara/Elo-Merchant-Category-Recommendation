import datetime
import feather
import gc
import pandas as pd
import numpy as np
import warnings

from utils import one_hot_encoder, save2pkl, loadpkl, reduce_mem_usage
from workalendar.america import Brazil

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(num_rows=None):
    # load csv & pkl
    hist_df = pd.read_csv('../input/historical_transactions.csv',nrows=num_rows)
    merchants_df = loadpkl('../features/merchants.pkl')

    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('A',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A':0, 'B':1, 'C':2})

    # category
    hist_df['category_1_approved'] = hist_df['category_1'] * hist_df['authorized_flag']
    hist_df['category_1_unapproved'] = hist_df['category_1'] * (1-hist_df['authorized_flag'])
    hist_df['category_2_approved'] = hist_df['category_2'] * hist_df['authorized_flag']
    hist_df['category_2_unapproved'] = hist_df['category_2'] * (1-hist_df['authorized_flag'])
    hist_df['category_3_approved'] = hist_df['category_3'] * hist_df['authorized_flag']
    hist_df['category_3_unapproved'] = hist_df['category_3'] * (1-hist_df['authorized_flag'])

    # purchase amount
    hist_df['purchase_amount_outlier'] = (hist_df['purchase_amount']>0.8).astype(int)
    hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))
    hist_df['purchase_amount'] = np.round(hist_df['purchase_amount'] / 0.00150265118 + 497.06,2)
    hist_df['purchase_amount_approved'] = hist_df['purchase_amount'] * hist_df['authorized_flag']
    hist_df['purchase_amount_unapproved'] = hist_df['purchase_amount'] * (1-hist_df['authorized_flag'])

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['year'] = hist_df['purchase_date'].dt.year
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

    # month diff
    hist_df['month_diff'] = ((pd.to_datetime('2018-03-01') - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']
    hist_df['month_diff_approved'] = hist_df['month_diff'] * hist_df['authorized_flag']
    hist_df['month_diff_unapproved'] = hist_df['month_diff'] * (1-hist_df['authorized_flag'])

    # month lag
    hist_df['month_lag_approved'] = hist_df['month_lag'] * hist_df['authorized_flag']
    hist_df['month_lag_unapproved'] = hist_df['month_lag'] * (1-hist_df['authorized_flag'])

    # installments
    hist_df['installments_approved'] = hist_df['installments'] * hist_df['authorized_flag']
    hist_df['installments_unapproved'] = hist_df['installments'] * (1-hist_df['authorized_flag'])

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

    # seasonality
    cal = Brazil()
    hist_df['is_holiday'] = hist_df['purchase_date'].dt.date.apply(cal.is_holiday).astype(int)

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

    # additional features
    hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
    hist_df['duration_approved'] = hist_df['duration']*hist_df['authorized_flag']
    hist_df['duration_unapproved'] = hist_df['duration']*(1-hist_df['authorized_flag'])
    hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']
    hist_df['amount_month_ratio_approved'] = hist_df['amount_month_ratio']*hist_df['authorized_flag']
    hist_df['amount_month_ratio_unapproved'] = hist_df['amount_month_ratio']*(1-hist_df['authorized_flag'])

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)
    merchants_df = reduce_mem_usage(merchants_df)

    # merge merchants_df
    hist_df = pd.merge(hist_df, merchants_df, on='merchant_id', how='outer')
    merchants_cols = merchants_df.columns.tolist()
    del merchants_df
    gc.collect()

    col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    for col in merchants_cols:
        aggs[col] = ['sum', 'mean']

    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
    aggs['purchase_amount_approved'] = ['sum','max','min','mean','var','skew']
    aggs['purchase_amount_unapproved'] = ['sum','max','min','mean','var','skew']
    aggs['purchase_amount_outlier'] = ['sum','mean']
    aggs['installments'] = ['sum','max','min','mean','var','skew']
    aggs['installments_approved'] = ['sum','max','min','mean','var','skew']
    aggs['installments_unapproved'] = ['sum','max','min','mean','var','skew']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['sum','max','min','mean','var','skew']
    aggs['month_lag_approved'] = ['sum','max','min','mean','var','skew']
    aggs['month_lag_unapproved'] = ['sum','max','min','mean','var','skew']
    aggs['month_diff'] = ['sum','max','min','mean','var','skew']
    aggs['month_diff_approved'] = ['sum','max','min','mean','var','skew']
    aggs['month_diff_unapproved'] = ['sum','max','min','mean','var','skew']
    aggs['authorized_flag'] = ['mean', 'sum']
    aggs['category_1'] = ['mean','min']
    aggs['category_1_approved'] = ['mean']
    aggs['category_1_unapproved'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_2_approved'] = ['mean']
    aggs['category_2_unapproved'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['category_3_approved'] = ['mean']
    aggs['category_3_unapproved'] = ['mean']
    aggs['card_id'] = ['size','count']
    aggs['is_holiday'] = ['mean']
    aggs['price'] = ['sum','max','min','mean','var','skew']
    aggs['purchase_amount_outlier']=['mean']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration']=['sum','max','min','mean','var','skew']
    aggs['duration_approved']=['sum','max','min','mean','var','skew']
    aggs['duration_unapproved']=['sum','max','min','mean','var','skew']
    aggs['amount_month_ratio']=['sum','max','min','mean','var','skew']
    aggs['amount_month_ratio_approved']=['sum','max','min','mean','var','skew']
    aggs['amount_month_ratio_unapproved']=['sum','max','min','mean','var','skew']

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
    hist_df['hist_purchase_date_uptonow'] = (pd.to_datetime('2019-01-01')-hist_df['hist_purchase_date_max']).dt.days
    hist_df['hist_purchase_date_uptomin'] = (pd.to_datetime('2019-01-01')-hist_df['hist_purchase_date_min']).dt.days

    # save
    save2pkl('../features/historical_transactions.pkl', hist_df)

if __name__ == '__main__':
    main()
