
import gc
import json
import numpy as np
import pandas as pd

from boruta import BorutaPy
from glob import glob
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from tqdm import tqdm

from utils import line_notify, FEATS_EXCLUDED

def main():
    # load feathers
    files = sorted(glob('../features/*.feather'))
    df = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)], axis=1)

    # set card_id as index
    df.set_index('card_id', inplace=True)

    # use selected features
    df = df[configs['features']]

    # split train & test
    train_df = df[df['target'].notnull()]
    del df
    gc.collect()

    # fill nan
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df.fillna(0, inplace=True)

    train_df = train_df[:10000] # debug

    # Lightgbm Regresssor
    lgbmclf = LGBMRegressor(boosting_type='rf',
                            objective='regression',
                            num_iteration=10000,
                            num_leaves=31,
                            min_data_in_leaf=27,
                            max_depth=-1,
                            learning_rate=0.015,
                            feature_fraction= 0.9,
                            bagging_freq= 1,
                            bagging_fraction= 0.9,
                            bagging_seed= 11,
                            metric= 'rmse',
                            lambda_l1=0.1,
                            verbosity= -1,
                            nthread= 4,
                            random_state= 4950)

    # define Boruta feature selection method
    feat_selector = BorutaPy(lgbmclf, n_estimators='auto', verbose=-1)
#    feat_selector = BorutaPy(lgbmclf)

    # features
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # fit
    feat_selector.fit(train_df[feats].values, train_df['target'].values)

    # check selected features - first 5 features are selected
    print(feat_selector.support_)

    # check ranking of features
    print(feat_selector.ranking_)

    # save feature name list
    features_json = {}
    features_json['features'] = train_df[feats].columns[feat_selector.support_].tolist()
    with open('../features/features_selected.json', 'w') as f:
        json.dump(features_json, f, indent=4)

if __name__ == '__main__':
    configs = json.load(open('../configs/200_all.json'))
    main()
