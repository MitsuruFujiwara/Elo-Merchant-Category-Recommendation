import gc
import numpy as np
import pandas as pd
import xgboost

from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, StratifiedKFold

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

xgb_train = xgboost.DMatrix(TRAIN_DF[FEATS],
                        np.log1p(TRAIN_DF['visitors'])
                        )

def xgb_eval(gamma,
             max_depth,
             min_child_weight,
             subsample,
             colsample_bytree,
             colsample_bylevel,
             alpha,
             _lambda):

    params = {
            'objective':'gpu:reg:linear', # GPU parameter
            'booster': 'gbtree',
            'eval_metric':'rmse',
            'silent':1,
            'eta': 0.02,
            'tree_method': 'gpu_hist', # GPU parameter
            'predictor': 'gpu_predictor', # GPU parameter
            'seed':326
            }

    params['gamma'] = gamma
    params['max_depth'] = int(max_depth)
    params['min_child_weight'] = min_child_weight
    params['subsample'] = max(min(subsample, 1), 0)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['colsample_bylevel'] = max(min(colsample_bylevel, 1), 0)
    params['alpha'] = max(alpha, 0)
    params['lambda'] = max(_lambda, 0)


#    folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=47)

    reg = xgboost.cv(params=params,
                     dtrain=xgb_train,
                     num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                     nfold=NUM_FOLDS,
                     metrics=["rmse"],
#                     folds=folds.split(TRAIN_DF[FEATS], TRAIN_DF['park_japanese_holiday']),
                     early_stopping_rounds=200,
                     verbose_eval=100,
                     seed=47,
                     )
    gc.collect()
    return -reg['test-rmse-mean'].iloc[-1]

def main():
    # reg for bayesian optimization
    reg_bo = BayesianOptimization(xgb_eval, {'gamma':(0, 1),
                                             'max_depth': (3, 8),
                                             'min_child_weight': (0, 45),
                                             'subsample': (0.001, 1),
                                             'colsample_bytree': (0.001, 1),
                                             'colsample_bylevel': (0.001, 1),
                                             'alpha': (9, 20),
                                             '_lambda': (0, 10)
                                             })

    reg_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(reg_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('../output/max_params_xgb.csv')

    line_notify('Bayes Opt XGBoost finished.')

if __name__ == '__main__':
    main()
