#!/bin/sh
python 106_train_xgb.py
python 107_train_lgbm.py
python 201_blend.py
