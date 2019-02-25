
import gc
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from glob import glob
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils import line_notify, NUM_FOLDS, FEATS_EXCLUDED, rmse, submit, to_feature

################################################################################
# Train LightGBM with non-outliers
################################################################################

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Display/plot feature importance
def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # for checking all importance
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train, test, num_folds, stratified = False, debug= False):

    # only use non-outlier
    train_df = train[train['outliers']==0]
    test_df = test

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=4950)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=4950)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params ={
                'device' : 'gpu',
                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'num_leaves': 47,
                'colsample_bytree': 0.287873914837844,
                'subsample': 0.401943798226545,
                'max_depth': 11,
                'reg_alpha': 8.48787854475937,
                'reg_lambda': 2.33680821212747,
                'min_split_gain': 2.45990086671046,
                'min_child_weight': 7.39278772239252,
                'min_data_in_leaf': 63,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        # train model
        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        reg.save_model('../output/lgbm_'+str(n_fold)+'_non_outlier.txt')

        # save predictions
        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        # save feature importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSE score and LINE Notify
    full_rmse = rmse(train_df['target'], oof_preds)
    line_notify('Full RMSE score %.6f' % full_rmse)

    # display importances
    display_importances(feature_importance_df,
                        '../output/lgbm_importances_non_outlier.png',
                        '../output/feature_importance_lgbm_non_outlier.csv')

    if not debug:
        # save prediction for submit
        test.loc[:,'target'] = sub_preds
        test = test.reset_index()
        test[['card_id', 'target']].to_csv(submission_file_name, index=False)

        # save out of fold prediction
        train.loc[train['outliers']==0,'OOF_PRED'] = oof_preds
        train = train.reset_index()
        train[['card_id', 'OOF_PRED']].to_csv(oof_file_name, index=False)

def main(debug=False):
    with timer("Load Datasets"):
        # load feathers
        files = sorted(glob('../features/*.feather'))
        df = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)], axis=1)

        # set card_id as index
        df.set_index('card_id', inplace=True)

        # use selected features
        df = df[configs['features']]

        # split train & test
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        del df
        gc.collect()
    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(train_df, test_df, num_folds=NUM_FOLDS, stratified=False, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission_lgbm_non_outlier.csv"
    oof_file_name = "../output/oof_lgbm_non_outlier.csv"
    configs = json.load(open('../configs/202_lgbm_non_outlier.json'))
    with timer("Full model run"):
        main(debug=False)
