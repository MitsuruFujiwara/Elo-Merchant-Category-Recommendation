
import gc
import json
import numpy as np
import optuna
import pandas as pd
import xgboost
import warnings

from glob import glob
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils import FEATS_EXCLUDED, NUM_FOLDS, loadpkl, line_notify

################################################################################
# optunaによるhyper parameter最適化
# 参考: https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
################################################################################

warnings.simplefilter(action='ignore', category=FutureWarning)

# load datasets
CONFIGS = json.load(open('../configs/205_xgb.json'))

# load feathers
FILES = sorted(glob('../features/*.feather'))
DF = pd.concat([pd.read_feather(f) for f in tqdm(FILES, mininterval=60)], axis=1)

# set card_id as index
DF.set_index('card_id', inplace=True)

# use selected features
DF = DF[CONFIGS['features']]

# split train & test
TRAIN_DF = DF[DF['target'].notnull()]
TEST_DF = DF[DF['target'].isnull()]

del DF, TEST_DF
gc.collect()

FEATS = [f for f in TRAIN_DF.columns if f not in FEATS_EXCLUDED]

def objective(trial):
    xgb_train = xgboost.DMatrix(TRAIN_DF[FEATS],
                                label=TRAIN_DF['target'])

    param = {
             'objective':'gpu:reg:linear', # GPU parameter
             'tree_method': 'gpu_hist', # GPU parameter
             'predictor': 'gpu_predictor', # GPU parameter
             'eval_metric':'rmse',
             'silent':1,
             'eta': 0.01,
             'booster': 'gbtree',
             'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
             'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
             }

    param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
    param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
    param['min_child_weight'] = trial.suggest_uniform('min_child_weight', 0, 45)
    param['subsample']=trial.suggest_uniform('subsample', 0.001, 1)
    param['colsample_bytree']=trial.suggest_uniform('colsample_bytree', 0.001, 1)
    param['colsample_bylevel'] = trial.suggest_uniform('colsample_bylevel', 0.001, 1)

    folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=4950)

    clf = xgboost.cv(params=param,
                     dtrain=xgb_train,
                     metrics=['rmse'],
                     nfold=NUM_FOLDS,
#                     stratified=True,
                     folds=list(folds.split(TRAIN_DF[FEATS], TRAIN_DF['outliers'])),
                     num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                     early_stopping_rounds=200,
                     verbose_eval=100,
                     seed=47
                     )
    gc.collect()
    return clf['test-rmse-mean'].iloc[-1]

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # save result
    hist_df = study.trials_dataframe()
    hist_df.to_csv("../output/optuna_result_xgb.csv")

    line_notify('optuna XGBoost finished.')
