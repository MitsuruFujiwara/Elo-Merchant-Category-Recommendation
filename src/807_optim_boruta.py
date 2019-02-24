
import gc
import json
import numpy as np
import pandas as pd

from boruta import BorutaPy
from glob import glob
#from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.feature_selection import RFECV
from tqdm import tqdm

from utils import line_notify, FEATS_EXCLUDED

################################################################################
# feature selection by boruta (not used)
################################################################################

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

    train_df = train_df[:10000] # debug

    # fill nan
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df.fillna(0, inplace=True)

    # RandomForest Regressor
    rfc = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=6)

    # define Boruta feature selection method
    print('Starting Boruta')
    feat_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, perc=100)
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
#    configs = json.load(open('../configs/207_lgbm_best.json'))
    main()
