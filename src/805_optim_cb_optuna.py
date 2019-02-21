
import catboost as cb
import gc
import json
import numpy as np
import optuna
import pandas as pd
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
CONFIGS = json.load(open('../configs/206_cb.json'))

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

    cb_train = cb.Pool(TRAIN_DF[FEATS], label=TRAIN_DF['target'])

    param = {
             'task_type' : 'GPU',
             'loss_function': 'RMSE',
             'custom_metric': 'RMSE',
             'eval_metric': 'RMSE',
             'learning_rate': 0.01,
             'train_dir':'../output/catboost_info_optuna'
             }

    param['bootstrap_type']=trial.suggest_categorical('bootstrap_type',['Poisson','Bayesian','Bernoulli','No'])
    param['l2_leaf_reg']=trial.suggest_int('l2_leaf_reg', 1, 50)
    param['max_depth'] = trial.suggest_int('max_depth', 1, 10)

    if param['bootstrap_type']=='Poisson' or param['bootstrap_type']=='Bernoulli':
        param['subsample']=trial.suggest_uniform('subsample', 0.001, 1)

    clf = cb.cv(pool=cb_train,
                params=param,
                num_boost_round=10000,
                nfold=NUM_FOLDS,
                seed=326,
                verbose_eval=100,
                early_stopping_rounds=200)

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
    hist_df.to_csv("../output/optuna_result_cb.csv")

    line_notify('optuna CatBoost finished.')
