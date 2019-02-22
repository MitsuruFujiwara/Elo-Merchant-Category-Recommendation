
import numpy as np
import pandas as pd
import gc

from utils import line_notify, loadpkl, rmse, submit

################################################################################
# Blend & Submit
################################################################################

# search best threshold
def getBestThreshold(act, pred):
    rmse_bst = rmse(act, pred)
    print('oof rmse: {:.10f}'.format(rmse_bst))
    for _q in np.arange(0, 0.005, 0.00001):
        _threshold = pred.quantile(_q)
#        _pred = pred.apply(lambda x: x if x > _threshold else -33.21928095)
        _pred = pred.apply(lambda x: x if x > _threshold else -15)
        _rmse = rmse(act, _pred)
        if _rmse < rmse_bst:
            rmse_bst = _rmse
            q = _q
            th = _threshold
        if _q % 0.0001==0:
            print("q: {:.4f}, th: {:.4f}, rmse: {:.10f}".format(_q, _threshold, _rmse))

    print("best q: {:.4f}, best th: {:.4f}, best rmse: {:.10f}".format(q, th, rmse_bst))

    return th

def main():
    # submitファイルをロード
    sub = pd.read_csv("../input/sample_submission.csv")
    sub_lgbm = pd.read_csv("../output/submission_lgbm.csv")
    sub_xgb = pd.read_csv("../output/submission_xgb.csv")
    sub_lgbm_non_outlier = pd.read_csv("../output/submission_lgbm_non_outlier.csv")
    sub_xgb_non_outlier = pd.read_csv("../output/submission_xgb_non_outlier.csv")

    # non outlier
    sub_non_outlier=pd.DataFrame()
    sub_non_outlier['target'] = 0.5*sub_lgbm_non_outlier['target']+0.5*sub_xgb_non_outlier['target']

    # カラム名を変更
    sub.columns =['card_id', 'target']

    # merge
    sub.loc[:,'target'] = 0.5*sub_lgbm['target']+0.5*sub_xgb['target']

    # post processing
    sub.loc[sub['target']>0,'target'] = sub_non_outlier.loc[sub['target']>0,'target']

    del sub_lgbm, sub_xgb, sub_lgbm_non_outlier
    gc.collect()

    # train_dfをロード
    train_df = loadpkl('../output/train_df.pkl')

    # out of foldの予測値をロード
    oof_lgbm = pd.read_csv("../output/oof_lgbm.csv")
    oof_xgb = pd.read_csv("../output/oof_xgb.csv")
    oof_lgbm_non_outlier = pd.read_csv("../output/oof_lgbm_non_outlier.csv")
    oof_xgb_non_outlier = pd.read_csv("../output/oof_xgb_non_outlier.csv")

    # non outlier
    oof_preds_non_outlier = pd.DataFrame()
    oof_preds_non_outlier['OOF_PRED'] = 0.5*oof_lgbm_non_outlier['OOF_PRED']+0.5*oof_xgb_non_outlier['OOF_PRED']

    oof_preds = pd.DataFrame()
    oof_preds['OOF_PRED'] = 0.5*oof_lgbm['OOF_PRED']+0.5*oof_xgb['OOF_PRED']
    oof_preds.loc[(train_df['outliers']==0)&(oof_preds['OOF_PRED']>0),'OOF_PRED'] = oof_preds_non_outlier.loc[oof_preds['OOF_PRED']>0,'OOF_PRED']

    # get best threshold
    th = getBestThreshold(train_df['target'], oof_preds['OOF_PRED'])
#    th = sub['target'].quantile(.0004)
#    sub.loc[:,'target']=sub['target'].apply(lambda x: x if x > th else -33.21928095)
    sub.loc[:,'target']=sub['target'].apply(lambda x: x if x > th else -15)
#    oof_preds=oof_preds.apply(lambda x: x if x > th else -33.21928095)
    oof_preds['OOF_PRED']=oof_preds['OOF_PRED'].apply(lambda x: x if x > th else -15)

    # local cv scoreを算出
    local_rmse = rmse(train_df['target'], oof_preds)

    # LINE通知
    line_notify('Blend Local RMSE score %.6f' % local_rmse)

    del oof_lgbm, oof_xgb
    gc.collect()

    # save submit file
    sub[['card_id', 'target']].to_csv(submission_file_name, index=False)

    # API経由でsubmit
    submit(submission_file_name, comment='model301 cv: %.6f' % local_rmse)

if __name__ == '__main__':
    submission_file_name = "../output/submission_blend.csv"
    main()
