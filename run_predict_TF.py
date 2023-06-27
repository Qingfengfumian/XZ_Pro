import numpy as np
import os
import xgboost as xgb
import datetime
import pandas as pd
from model import build_emb_model,build_xgb_model,build_lgb_model,load_model,evaluate_model_result
from utils import mkdir,real_warns_4G,load_model_infos, save_model_infos, load_from_json,find_new_model_file
from data_proc_TF_4G import load_data,gen_emb_series,gen_model_data

def predict_TF(para, pred_ft_data, ft_cols,distr):
    pred_input = pred_ft_data[ft_cols]
    pred_output1 = np.zeros(shape=[pred_input.shape[0]])
    pred_output2 = np.zeros(shape=[pred_input.shape[0]])
    if para.use_weighted_avg:
        xgbm_weights = load_from_json(os.path.join(para.model_path, "lgb_model-weights.json"))
        lgbm_weights = load_from_json(os.path.join(para.model_path, "xgb_model-weights.json"))
    else:
        xgbm_weights = {ii:1 for ii in range(1,6)}
        lgbm_weights = {ii:1 for ii in range(1,6)}
    xgbm_weights = {str(k): v for k, v in xgbm_weights.items()}
    lgbm_weights = {str(k): v for k, v in lgbm_weights.items()}
    mod_path = r"./Data/{}/Inter_Result/{}_退服_模型/".format(distr,para.ftype)
    new_path = find_new_model_file(mod_path,'xgb')
    model_date = new_path.split('_')[-1].split('.')[0]
    for i in range(1, 6):
        xgb_model_path = os.path.join(mod_path, "xgb_model_{}_{}.pkl".format(i,model_date))
        lgb_model_path = os.path.join(mod_path, "lgb_model_{}_{}.pkl".format(i,model_date))
        xgbmd = load_model(xgb_model_path)
        lgbmd = load_model(lgb_model_path)
        try:
            pred_output1 += xgbmd.predict_proba(pred_input, ntree_limit=xgbmd.best_iteration)[:, 1]*xgbm_weights[str(i)]
        except:
            pred_output1 += xgbmd.predict(xgb.DMatrix(pred_input, label=None), ntree_limit=xgbmd.best_iteration) * xgbm_weights[str(i)]
        pred_output2 += lgbmd.predict(pred_input, num_iteration=lgbmd.best_iteration)*lgbm_weights[str(i)]
    xgb_weight_sum = np.sum([value for value in xgbm_weights.values()])
    lgb_weight_sum = np.sum([value for value in lgbm_weights.values()])
    pred_output_prob = (pred_output1 / xgb_weight_sum + pred_output2 / lgb_weight_sum) / 2
    # if distr == '铜川':
    #     pred_output_label = [int(v >= (para.threshold+0.1)) for v in pred_output_prob]
    # else:
    #     pred_output_label = [int(v >= para.threshold) for v in pred_output_prob]
    pred_ft_data["pred_probability"] = pred_output_prob
    # pred_ft_data["pred_label"] = pred_output_label

    date = para.date
    origin_data = pd.read_csv(
            './Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_delJZ.csv'.format(distr, para.mode, para.ftype, date),
            encoding='gbk', engine='python')  # index_col=0)
    if "制式" in origin_data.columns:
        origin_data.drop("制式", axis=1, inplace=True)

    # test_data = origin_data.copy()
    test_data = origin_data.drop_duplicates(subset=['告警开始时间', '告警名称', '基站id'])
    test_data['date'] = pd.to_datetime(test_data['告警开始时间']).dt.date
    test_data['告警开始时间'] = pd.to_datetime(test_data['告警开始时间'], format="%Y-%m-%d %H:%M:%S")
    test_data['基站id'] = test_data['基站id'].map(lambda x: str(x).split('.')[0])
    tf_data = test_data[test_data['告警名称'].isin(real_warns_4G)]
    nighttf_data = tf_data.copy()
    nighttf_data["data_hour"] = pd.to_datetime(nighttf_data['告警开始时间']).dt.hour
    nighttf_data = nighttf_data.loc[
        ((nighttf_data["data_hour"] <= 6) | (nighttf_data["data_hour"] >= 23))]  # 小

    ff = load_from_json(os.path.join(mod_path, "FRule_{}_20200701-20210228.txt".format(distr)))

    def find_info(x, y):
        iid = str(int(x))
        ddate = str(y).split()[0]  # 参数格式是timestamp
        ddate = datetime.datetime.date(datetime.datetime.strptime(ddate, '%Y-%m-%d'))  # 参数是datetime.datetime
        warn_cnt = test_data[(test_data['基站id'] == iid)].shape[0]
        night_data = tf_data[(tf_data['基站id'] == iid)]
        nighttf_data1 = nighttf_data[(nighttf_data['基站id'] == iid)]
        night_data.sort_values(by='告警开始时间', ascending=False, inplace=True)
        if night_data.shape[0] == 0:
            last_w = 0
        else:
            last_w = night_data['告警开始时间'].tolist()[0]

        b1_data = test_data[
            (test_data['告警开始时间'] >= ddate - datetime.timedelta(days=1)) & (test_data['告警开始时间'] < ddate) & (test_data['基站id'] == iid)]
        b2_data = test_data[
            (test_data['告警开始时间'] >= ddate - datetime.timedelta(days=2)) & (test_data['告警开始时间'] < ddate) & (test_data['基站id'] == iid)]

        tf1_data = tf_data[
            (tf_data['告警开始时间'] >= ddate - datetime.timedelta(days=1)) & (tf_data['告警开始时间'] < ddate) & (tf_data['基站id'] == iid)]
        b2_warn = b2_data['告警名称'].tolist()
        conf = 0
        conf_info = []
        for n, f in enumerate(ff):
            if sum([1 if x in b2_warn else 0 for x in f[:-1]]) == (len(f) - 1):
                # if set(f[:-1])<=b2_warn:
                conf += (float(f[-1]))
                conf_info.append(str(f[-1]))
            else:
                conf_info.append('0')
        return pd.Series([warn_cnt, night_data.shape[0], len(set(night_data['date'])), len(set(nighttf_data1['date'])),
             last_w,b1_data.shape[0], tf1_data.shape[0], conf])#, conf_info

    print("统计输入数据前一周退服详情")
    from tqdm import tqdm
    # tqdm.pandas(desc='pandas bar')
    # pred_ft_data[['前七天告警数量', '前七天退服告警数量', '前七天退服天数', '前七天夜间退服天数','退服详情']] = pred_ft_data.progress_apply(lambda x: find_info(x['基站id'], x['date']), axis=1)
    pred_ft_data[['前七天告警数量', '前七天退服告警数量', '前七天退服天数', '前七天夜间退服天数','最近一次退服时间',
                   '前一天的告警数量','前一天退服告警数量','规则置信度sum']] = pred_ft_data.apply(lambda x: find_info(x['基站id'], x['date']), axis=1) # ,'规则置信度list'

    submit_data = pred_ft_data[["基站id", "date", "pred_probability",'前七天告警数量', '前七天退服告警数量', '前七天退服天数', '前七天夜间退服天数','最近一次退服时间',
                   '前一天的告警数量','前一天退服告警数量','规则置信度sum']] # ,'规则置信度list'
    # submit_data = pred_ft_data
    submit_data.sort_values(by=["基站id", "date"], ascending=True, inplace=True)
    if os.path.exists(r"./Data/{}/Inspect_List/{}/".format(distr, para.ftype)) == False:
        mkdir(r"./Data/{}/Inspect_List/{}/".format(distr, para.ftype))
    result_path = r"./Data/{}/Inspect_List/{}/TFPre_{}.csv".\
        format(distr, para.ftype, para.date)
    submit_data.to_csv(result_path, index=False, encoding="gbk")
    print("退服概率预测完成，结果保存在: {}。".format(result_path))
