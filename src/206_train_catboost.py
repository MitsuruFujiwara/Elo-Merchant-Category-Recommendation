
import catboost as cb
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from glob import glob
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils import line_notify, NUM_FOLDS, FEATS_EXCLUDED, rmse, submit

################################################################################
# Train CatBoost with outliers (not used)
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
    plt.title('CatBoost Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# CatBoost with KFold or Stratified KFold
def kfold_catboost(train_df, test_df, num_folds, stratified = False, debug= False):
    print("Starting CatBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

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
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        cb_train = cb.Pool(train_x, label=train_y)
        cb_test = cb.Pool(valid_x, label=valid_y)

        # params
        params ={
                'task_type' : 'GPU',
                'loss_function': 'RMSE',
                'custom_metric': 'RMSE',
                'eval_metric': 'RMSE',
                'learning_rate': 0.01,
                'early_stopping_rounds':200,
                'verbose_eval':100,
                'train_dir':'../output/catboost_info',
                'random_seed': int(2**n_fold)
                }

        reg = cb.train(
                       cb_train,
                       params,
                       num_boost_round=10000,
                       eval_set=cb_test
                       )

        # save model
        reg.save_model('../output/cb_'+str(n_fold)+'.txt')

        oof_preds[valid_idx] = reg.predict(valid_x)
        sub_preds += reg.predict(test_df[feats]) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importances_)
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_rmse = rmse(train_df['target'], oof_preds)
    line_notify('Full RMSE score %.6f' % full_rmse)

    # display importances
    display_importances(feature_importance_df,
                        '../output/cb_importances.png',
                        '../output/feature_importance_cb.csv')

    if not debug:
        # 提出データの予測値を保存
        test_df.loc[:,'target'] = sub_preds
        test_df = test_df.reset_index()

        # targetが一定値以下のものをoutlierで埋める
#        q_test = test_df['target'].quantile(.0005)
#        test_df.loc[:,'target']=test_df['target'].apply(lambda x: x if x > q_test else -33.21928095)
        test_df[['card_id', 'target']].to_csv(submission_file_name, index=False)

        # out of foldの予測値を保存
        train_df.loc[:,'OOF_PRED'] = oof_preds
        train_df = train_df.reset_index()

        # targetが一定値以下のものをoutlierで埋める
#        q_train = train_df['OOF_PRED'].quantile(.0005)
#        train_df.loc[:,'OOF_PRED'] = train_df['OOF_PRED'].apply(lambda x: x if x > q_train else -33.21928095)
        train_df[['card_id', 'OOF_PRED']].to_csv(oof_file_name, index=False)

        # Adjusted Full RMSEスコアの表示 & LINE通知
#        full_rmse_adj = rmse(train_df['target'], train_df['OOF_PRED'])
#        line_notify('Adjusted Full RMSE score %.6f' % full_rmse_adj)

        # API経由でsubmit
#        submit(submission_file_name, comment='model206 cv: %.6f' % full_rmse)

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
    with timer("Run CatBoost with kfold"):
        kfold_catboost(train_df, test_df, num_folds=NUM_FOLDS, stratified=False, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission_cb.csv"
    oof_file_name = "../output/oof_cb.csv"
    configs = json.load(open('../configs/206_cb.json'))
    with timer("Full model run"):
        main(debug=False)
