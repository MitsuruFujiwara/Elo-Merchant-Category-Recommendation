
import numpy as np
import pandas as pd
import gc

from utils import line_notify, loadpkl, rmse, submit

################################################################################
# Blend & Submit
################################################################################

def main():
    # submitファイルをロード
    sub = pd.read_csv("../input/sample_submission.csv")
    sub_lgbm = pd.read_csv("../output/submission_lgbm.csv")
    sub_xgb = pd.read_csv("../output/submission_xgb.csv")

    # カラム名を変更
    sub.columns =['card_id', 'target']
    sub_lgbm.columns =['card_id', 'target']
    sub_xgb.columns =['card_id', 'target']

    # merge
    sub.loc[:,'target'] = 0.5*sub_lgbm['target']+0.5*sub_xgb['target']

    del sub_lgbm, sub_xgb
    gc.collect()

    # out of foldの予測値をロード
    oof_lgbm = pd.read_csv("../output/oof_lgbm.csv")
    oof_xgb = pd.read_csv("../output/oof_xgb.csv")
    oof_preds = 0.5*oof_lgbm['OOF_PRED']+0.5*oof_xgb['OOF_PRED']

    # train_dfをロード
    train_df = loadpkl('../output/train_df.pkl')

    # local cv scoreを算出
    local_rmse = rmse(train_df['target'], oof_preds)

    # LINE通知
    line_notify('Blend Local RMSE score %.6f' % local_rmse)

    del oof_lgbm, oof_xgb
    gc.collect()

    # save submit file
    sub[['card_id', 'target']].to_csv(submission_file_name, index=False)

    # API経由でsubmit
    submit(submission_file_name, comment='model201 cv: %.6f' % local_rmse)

if __name__ == '__main__':
    submission_file_name = "../output/submission_blend.csv"
    main()
