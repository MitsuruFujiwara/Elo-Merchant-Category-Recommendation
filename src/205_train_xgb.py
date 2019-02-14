
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import warnings
import xgboost as xgb

from contextlib import contextmanager
from glob import glob
from sklearn.model_selection import KFold, StratifiedKFold
from pandas.core.common import SettingWithCopyWarning
from tqdm import tqdm

from utils import line_notify, NUM_FOLDS, FEATS_EXCLUDED, rmse, submit

################################################################################
# Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
# 学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
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
    plt.title('XGBoost Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# XGBoost GBDT with KFold or Stratified KFold
def kfold_xgboost(train_df, test_df, num_folds, stratified = False, debug= False):
    print("Starting XGBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

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

    # final predict用にdmatrix形式のtest dfを作っておきます
    test_df_dmtrx = xgb.DMatrix(test_df[feats])

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        xgb_train = xgb.DMatrix(train_x,
                                label=train_y)
        xgb_test = xgb.DMatrix(valid_x,
                               label=valid_y)

        # params
        params = {
                'objective':'gpu:reg:linear', # GPU parameter
                'booster': 'gbtree',
                'eval_metric':'rmse',
                'silent':1,
                'eta': 0.01,
                'max_leaves': 63,
                'colsample_bytree': 0.5665320670155495,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'reg_alpha': 9.677537745007898,
                'reg_lambda': 8.2532317400459,
                'gamma': 9.820197773625843,
                'min_child_weight': 41.9612869171337,
                'tree_method': 'gpu_hist', # GPU parameter
                'predictor': 'gpu_predictor', # GPU parameter
                'seed':int(2**n_fold)
                }

        reg = xgb.train(
                        params,
                        xgb_train,
                        num_boost_round=10000,
                        evals=[(xgb_train,'train'),(xgb_test,'test')],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        reg.save_model('../output/xgb_'+str(n_fold)+'.txt')

        oof_preds[valid_idx] = reg.predict(xgb_test)
        sub_preds += reg.predict(test_df_dmtrx) / folds.n_splits

        fold_importance_df = pd.DataFrame.from_dict(reg.get_score(importance_type='gain'), orient='index', columns=['importance'])
        fold_importance_df["feature"] = fold_importance_df.index.tolist()
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
                        '../output/xgb_importances.png',
                        '../output/feature_importance_xgb.csv')

    if not debug:
        # 提出データの予測値を保存
        test_df.loc[:,'target'] = sub_preds
        test_df = test_df.reset_index()

        # targetが一定値以下のものをoutlierで埋める
#        q_test = test_df['target'].quantile(.0006)
#        test_df.loc[:,'target']=test_df['target'].apply(lambda x: x if x > q_test else -33.21928095)
        test_df[['card_id', 'target']].to_csv(submission_file_name, index=False)

        # out of foldの予測値を保存
        train_df.loc[:,'OOF_PRED'] = oof_preds
        train_df = train_df.reset_index()

        # targetが一定値以下のものをoutlierで埋める
#        q_train = train_df['OOF_PRED'].quantile(.0006)
#        train_df.loc[:,'OOF_PRED'] = train_df['OOF_PRED'].apply(lambda x: x if x > q_train else -33.21928095)
        train_df[['card_id', 'OOF_PRED']].to_csv(oof_file_name, index=False)

        # Adjusted Full RMSEスコアの表示 & LINE通知
#        full_rmse_adj = rmse(train_df['target'], train_df['OOF_PRED'])
#        line_notify('Adjusted Full RMSE score %.6f' % full_rmse_adj)

        # API経由でsubmit
        submit(submission_file_name, comment='model205 cv: %.6f' % full_rmse)

def main(debug=False, use_pkl=False):
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
    with timer("Run XGBoost with kfold"):
        kfold_xgboost(train_df, test_df, num_folds=NUM_FOLDS, stratified=False, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission_xgb.csv"
    oof_file_name = "../output/oof_xgb.csv"
    configs = json.load(open('../configs/205_xgb.json'))
    with timer("Full model run"):
        main(debug=False,use_pkl=True)
