
import xgboost
import numpy as np
import pandas as pd
import optuna
import gc

from sklearn.model_selection import KFold, StratifiedKFold

from utils import FEATS_EXCLUDED, NUM_FOLDS, loadpkl, line_notify

################################################################################
# optunaによるhyper parameter最適化
# 参考: https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
################################################################################

# load datasets
TRAIN_DF = loadpkl('../output/train_df.pkl')
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
             'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
             'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
             'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
             }

    if param['booster'] == 'gbtree' or param['booster'] == 'dart':
        param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
        param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
        param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
#        param['min_child_weight'] = trial.suggest_uniform('min_child_weight', 0, 45),
#        param['subsample'] = trial.suggest_uniform('subsample', 0.001, 1),
#        param['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0.001, 1),
#        param['colsample_bylevel'] = trial.suggest_uniform('colsample_bylevel', 0.001, 1),
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

#    folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=4950)
#    print(folds.split(TRAIN_DF[FEATS], TRAIN_DF['outliers']))
    clf = xgboost.cv(params=param,
                     dtrain=xgb_train,
                     metrics=['rmse'],
                     nfold=NUM_FOLDS,
#                     stratified=True,
#                     folds=folds.split(TRAIN_DF[FEATS], TRAIN_DF['outliers']),
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
