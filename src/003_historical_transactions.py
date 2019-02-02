import datetime
import feather
import gc
import pandas as pd
import numpy as np
import warnings

from utils import one_hot_encoder

warnings.simplefilter(action='ignore', category=FutureWarning)

# numeric features
def hist_num():
    # load csv
    hist_df = pd.read_csv('../input/historical_transactions.csv')

    # fillna
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # trim
    hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

# categorical features
def hist_cat():
    # load csv
    hist_df = pd.read_csv('../input/historical_transactions.csv')

    # fillna
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('A',inplace=True)

# seasonality features
def hist_seas():
    # load csv
    hist_df = pd.read_csv('../input/historical_transactions.csv')

    # to datetime
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])

# additional features
def hist_add():
    


def main():


if __name__ == '__main__':
    main()
