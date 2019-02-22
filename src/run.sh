#!/bin/sh
#python 001_train_test.py
#python 002_merchants.py
#python 003_historical_transactions.py
#python 004_new_merchant_transactions.py
#python 101_aggregation.py
#python 201_train_lgbm.py
python 202_train_lgbm_non_outlier.py
python 203_train_xgb.py
python 204_train_xgb_non_outlier.py
python 301_blend.py
#python 801_optim_lgbm_optuna.py
#python 802_optim_lgbm_optuna_non_outlier.py
#python 803_optim_xgb_optuna.py
#python 804_optim_xgb_optuna_non_outlier.py
#python 805_optim_cb_optuna.py
