import datetime
import feather
import gc
import json
import pandas as pd
import numpy as np
import warnings

from utils import one_hot_encoder, loadpkl, to_feature

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
    df['card_id_cnt_ratio'] = df['new_card_id_count']/df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max']+df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min']+df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum']/df['hist_purchase_amount_sum']
    df['month_diff_mean'] = df['new_month_diff_mean']+df['hist_month_diff_mean']
    df['month_diff_max'] = df['new_month_diff_max']+df['hist_month_diff_max']
#    df['month_diff_min'] = df['new_month_diff_min']+df['hist_month_diff_min']
    df['month_lag_mean'] = df['new_month_lag_mean']+df['hist_month_lag_mean']
#    df['month_lag_max'] = df['new_month_lag_max']+df['hist_month_lag_max']
#    df['month_lag_min'] = df['new_month_lag_min']+df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean']+df['hist_category_1_mean']
    df['category_2_mean'] = df['new_category_2_mean']+df['hist_category_2_mean']
    df['category_3_mean'] = df['new_category_3_mean']+df['hist_category_3_mean']
#    df['category_1_min'] = df['new_category_1_min']+df['hist_category_1_min']
    df['installments_total'] = df['new_installments_sum']+df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean']+df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max']+df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum']/df['hist_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean'] = df['new_duration_mean']+df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min']+df['hist_duration_min']
    df['duration_max'] = df['new_duration_max']+df['hist_duration_max']
    df['amount_month_ratio_mean']=df['new_amount_month_ratio_mean']+df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min']=df['new_amount_month_ratio_min']+df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max']=df['new_amount_month_ratio_max']+df['hist_amount_month_ratio_max']
    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

    return df

def main():
    # load pkls
    df = loadpkl('../features/train_test.pkl')
    hist = loadpkl('../features/historical_transactions.pkl')
    new = loadpkl('../features/new_merchant_transactions.pkl')

    # merge
    df = pd.merge(df, hist, on='card_id', how='outer')
    df = pd.merge(df, new, on='card_id', how='outer')

    del hist, new
    gc.collect()

    # additional features
    df = additional_features(df)

    # save as feather
    to_feature(df, '../features')

    # save feature name list
    features_json = {}
    features_json['features'] = df.columns.tolist()
    with open('../features/all_features.json', 'w') as f:
        json.dump(features_json, f, indent=4)

if __name__ == '__main__':
    main()