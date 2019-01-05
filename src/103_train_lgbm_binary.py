
import gc
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from pandas.core.common import SettingWithCopyWarning

from preprocessing_002 import train_test, historical_transactions, merchants, new_merchant_transactions, additional_features
from utils import line_notify, NUM_FOLDS, FEATS_EXCLUDED, loadpkl, save2pkl, rmse, submit

################################################################################
# Model For Outliers Classification
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

    # importance下位の確認用に追加しました
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['outliers'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['outliers'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # パラメータは適当です
        params ={
                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.01,
#                'num_leaves': 32,
#                'colsample_bytree': 0.20461151519044,
#                'subsample': 0.805742797052828,
#                'max_depth': 10,
#                'reg_alpha': 0.196466392224054,
#                'reg_lambda': 0.045887453950229,
#                'min_split_gain': 0.247050274075659,
#                'min_child_weight': 23.9202696807894,
#                'min_data_in_leaf': 24,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        clf = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        clf.save_model('../output/lgbm_'+str(n_fold)+'_binary.txt')

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(clf.feature_importance(importance_type='gain', iteration=clf.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_auc = roc_auc_score(train_df['outliers'], oof_preds)
    line_notify('Full AUC score %.6f' % full_auc)

    # display importances
    display_importances(feature_importance_df,
                        '../output/lgbm_importances_binary.png',
                        '../output/feature_importance_lgbm_binary.csv')

    if not debug:
        # 提出データの予測値を保存
        test_df.loc[:,'Outlier_Likelyhood'] = sub_preds
        q_test = test_df['Outlier_Likelyhood'].quantile(.98907) # 1.0930%
        test_df.loc[:,'outliers']=test_df['Outlier_Likelyhood'].apply(lambda x: 1 if x > q_test else 0)
        test_df.loc[test_df['outliers']==1,'target']=-33.21928095

        # out of foldの予測値を保存
        train_df.loc[:,'Outlier_Likelyhood'] = oof_preds

        # save pkl
        save2pkl('../output/train_df.pkl', train_df)
        save2pkl('../output/test_df.pkl', test_df)

def main(debug=False, use_pkl=False):
    num_rows = 10000 if debug else None
    if use_pkl:
        df = loadpkl('../output/df.pkl')
    else:
        with timer("train & test"):
            df = train_test(num_rows)
        with timer("merchants"):
            merchants_df = merchants(num_rows=num_rows)
        with timer("historical transactions"):
            df = pd.merge(df, historical_transactions(merchants_df, num_rows), on='card_id', how='outer')
        with timer("new merchants"):
            df = pd.merge(df, new_merchant_transactions(merchants_df, num_rows), on='card_id', how='outer')
            del merchants_df
            gc.collect()
        with timer("additional features"):
            df = additional_features(df)
        with timer("save pkl"):
            save2pkl('../output/df.pkl', df)
    with timer("Run LightGBM with kfold"):
        print("df shape:", df.shape)
        kfold_lightgbm(df, num_folds=NUM_FOLDS, stratified=True, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission_binary.csv"
    oof_file_name = "../output/oof_lgbm_binary.csv"
    with timer("Full model run"):
        main(debug=False,use_pkl=True)
