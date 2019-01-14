import gc
import numpy as np
import pandas as pd
import lightgbm

from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, StratifiedKFold

from preprocess import train_test, nightley, hotlink, colopl, weather, nied_oyama, jorudan, agoop
from utils import FEATS_EXCLUDED, NUM_FOLDS, loadpkl, line_notify

# 以下参考
# https://github.com/fmfn/BayesianOptimization
# https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

NUM_ROWS=None
USE_PKL=True

if USE_PKL:
    DF = loadpkl('../output/df.pkl')
else:
    DF = train_test(NUM_ROWS)
    DF = pd.merge(DF, nightley(NUM_ROWS), on=['datetime', 'park'], how='outer')
    DF = pd.merge(DF, hotlink(NUM_ROWS), on='datetime', how='outer')
    DF = pd.merge(DF, colopl(NUM_ROWS), on=['year','month'], how='outer')
    DF = pd.merge(DF, weather(NUM_ROWS), on=['datetime', 'park'], how='outer')
    DF = pd.merge(DF, nied_oyama(NUM_ROWS), on=['datetime', 'park'], how='outer')
    DF = pd.merge(DF, agoop(num_rows), on=['park', 'year','month'], how='outer')
    DF = pd.merge(DF, jorudan(num_rows), on=['datetime', 'park'], how='outer')

# split test & train
TRAIN_DF = DF[DF['visitors'].notnull()]
FEATS = [f for f in TRAIN_DF.columns if f not in FEATS_EXCLUDED]

lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                              np.log1p(TRAIN_DF['visitors']),
                              free_raw_data=False
                              )

def lgbm_eval(num_leaves,
              colsample_bytree,
              subsample,
              max_depth,
              reg_alpha,
              reg_lambda,
              min_split_gain,
              min_child_weight,
              min_data_in_leaf
              ):

    params = dict()
    params["learning_rate"] = 0.01
#    params["silent"] = True
    params['device'] = 'gpu'
#    params["nthread"] = 16
    params['objective'] = 'regression'
    params['seed']=326,

    params["num_leaves"] = int(num_leaves)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)
    params['max_depth'] = int(max_depth)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    params['min_data_in_leaf'] = int(min_data_in_leaf)
    params['verbose']=-1

    folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=47)

    clf = lightgbm.cv(params=params,
                      train_set=lgbm_train,
                      metrics=['rmse'],
                      nfold=NUM_FOLDS,
                      folds=folds.split(TRAIN_DF[FEATS], TRAIN_DF['park_japanese_holiday']),
                      num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                      early_stopping_rounds=200,
                      verbose_eval=100,
                      seed=47,
                     )
    gc.collect()
    return -clf['rmse-mean'][-1]

def main():

    # clf for bayesian optimization
    clf_bo = BayesianOptimization(lgbm_eval, {'num_leaves': (16, 64),
                                              'colsample_bytree': (0.001, 1),
                                              'subsample': (0.001, 1),
                                              'max_depth': (8, 16),
                                              'reg_alpha': (0, 10),
                                              'reg_lambda': (0, 10),
                                              'min_split_gain': (0, 1),
                                              'min_child_weight': (0, 45),
                                              'min_data_in_leaf': (0, 500),
                                              })

    clf_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(clf_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('../output/max_params_lgbm.csv')

    line_notify('Bayes Opt LightGBM finished.')

if __name__ == '__main__':
    main()
