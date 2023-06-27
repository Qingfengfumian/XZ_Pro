import argparse
import os
import multiprocessing
import time
import pandas as pd
from xgboost import XGBClassifier
import lightgbm as lgb
from gensim.models import word2vec
from sklearn.model_selection import train_test_split, GroupKFold
import numpy as np
from matplotlib import pyplot
import joblib
import logging
import copy
import json
import math
import multiprocessing
import datetime
import random
from multiprocessing import Process, Manager
from collections import deque
from numpy.lib.stride_tricks import as_strided as stride
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# real_warns = ["eNodeB退服告警","基站退服","MME衍生基站退服","小区不可用告警","网元连接中断"]
# real_warns = ["MME衍生基站退服","LTE小区退出服务","MME衍生小区退服","基站退服","网元断链告警"]


# model_dir = './materials'
data_dir = './'
# material_dir = './materials'
# save_path = './'


def params_setup():
    """
    模型运行相关参数配置生成
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=7)  # 输入步长(天数)
    parser.add_argument('--target_long', type=int, default=3)  # 预测天数
    parser.add_argument('--day_ft_num', type=int, default=300)  # 每天取多少个告警特征
    parser.add_argument('--lgb_lr', type=float, default=0.10)  # lgb模型参数
    parser.add_argument('--lgb_max_depth', type=int, default=5)
    parser.add_argument('--lgb_num_leaves', type=int, default=100)
    parser.add_argument('--xgb_lr', type=float, default=0.30)  # xgb模型参数
    parser.add_argument('--xgb_max_depth', type=int, default=5)
    parser.add_argument('--xgb_num_leaves', type=int, default=32)
    parser.add_argument('--xgb_show_loss', type=bool, default=True)
    parser.add_argument('--drop_hour_duplicate', type=bool, default=False)  # embedding 模型参数
    parser.add_argument('--emb_size', type=int, default=50)
    parser.add_argument('--emd_exist', type=bool, default=True)
    parser.add_argument('--emb_window', type=int, default=7)
    parser.add_argument('--emb_iter', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.70)  # 分类阈值
    parser.add_argument('--data_path', type=str,
                        default="./data/XianYang/train-0701-1231")  # HanZhong/TongChuan 输入数据文件路径
    parser.add_argument('--data_exist', type=bool, default=False)  # 是否直接使用已生成数据集
    parser.add_argument('--log_path', type=str, default="./logs/model4JZ@HZ0120-0103ALL.txt")  # log日志保存路径
    parser.add_argument('--log_write', type=str, default="a")  # log日志保存路径
    parser.add_argument('--result_path', type=str,
                        default="./model_result/predict_model4JZ@XY0701-1130.csv")  # 预测结果保存路径
    parser.add_argument('--mode', type=str, default="evaluate")  # 运行模式: "train", "train_eval", "evaluate", "predict"
    parser.add_argument('--test_size', type=float, default=0.20)  # 测试数据集比例
    parser.add_argument('--model_id', type=str, default="LN_20210217-20210731_Del7D")  # 模型唯一标识
    parser.add_argument('--ft_n_job', type=int, default=multiprocessing.cpu_count())  # 特征工程采用多少个线程
    #     para = parser.parse_args()
    para = parser.parse_args(args=[])
    return para


para = params_setup()


def generate_ft_data(gwdata, start_date0, date_data, ft_nday=7):
    start_pred_date = start_date0 + datetime.timedelta(days=ft_nday)
    gawdata = pd.merge(date_data, gwdata, on="data_date", how="left")
    ft_cols = [i for i in range(24)]
    gftdata = gawdata[["data_date"] + ft_cols].copy()
    gftdata.sort_values(by="data_date", ascending=True, inplace=True)
    gftdata["data_date"] = gawdata["data_date"] + datetime.timedelta(days=1)  # 锚定预测日期第一天
    gftdata.rename(columns={i: "date-1_{}".format(str(i)) for i in ft_cols}, inplace=True)
    if ft_nday > 1:
        for i in range(2, ft_nday + 1):
            gftdata[["date-{}_{}".format(str(i), str(ii)) for ii in ft_cols]] = gawdata[ft_cols].shift(i - 1)
    gftdata = gftdata.loc[gftdata["data_date"] >= start_pred_date]
    gftdata.dropna(axis=0, subset=gftdata.columns[1:], how="all", inplace=True)  # 去除冗余数据
    return gftdata


# 生成日期对应的标签
def get_pos_neg_dates(para, sdata, all_dates, real_warns, pred_day=3):
    true_label_data = pd.DataFrame(columns=["基站id", "date", "true_label"])
    for grp in sdata.groupby("基站id"):
        df = grp[1]
        df_pos = df[df["告警名称"].isin(real_warns)]
        pos_dates_list = df_pos["data_date"].tolist()
        pos_dates = list(set(pos_dates_list))
        if pred_day > 1:  # 预测多天增加正样本日期
            new_pos_dates = []
            for dn in range(1, pred_day):
                for pos_date in pos_dates:
                    new_pos_dates.append(pos_date - datetime.timedelta(days=dn))
            pos_dates = pos_dates + new_pos_dates
            pos_dates = list(set(pos_dates))
        neg_dates = [day for day in all_dates if not day in pos_dates]
        max_date = max(all_dates)  # max_date = df["date"].max() (?)   # 要做边界检查
        max_label_date = max_date - datetime.timedelta(days=pred_day) + 1  # 后面的不能说它一定为负, 修正: 应该+2
        if para.mode in ["predict", "evaluate"]:
            pos_dates = [pos_date for pos_date in pos_dates if pos_date <= max_label_date]  # 预测时正负样本统一
        neg_dates = [neg_date for neg_date in neg_dates if neg_date <= max_label_date]

        if len(pos_dates) > 0:
            true_label_data = true_label_data.append(pd.DataFrame({"基站id": [grp[0]] * len(pos_dates), "date": pos_dates,
                                                                   "true_label": [1] * len(pos_dates)}),
                                                     ignore_index=True)
        if len(neg_dates) > 0:
            true_label_data = true_label_data.append(pd.DataFrame({"基站id": [grp[0]] * len(neg_dates), "date": neg_dates,
                                                                   "true_label": [0] * len(neg_dates)}),
                                                     ignore_index=True)
    return true_label_data


def load_model_infos(file_path):
    list_info = []
    with open(file_path, "r") as f:
        for item in f.readlines():
            list_info.append(item.replace("\n", ""))
    return list_info


def save_model_infos(list_info, file_path):
    with open(file_path, "w") as f:
        for item in list_info:
            f.write(item + "\n")


def gen_model_data(para, source_data, warn_scores, real_warns,Org_path):
    start = time.clock()
    source_data["告警label"] = source_data["告警名称"].apply(lambda x: warn_scores[x])
    source_data1 = source_data[["告警label", "告警名称", "告警开始时间", "基站id", "data_date", "data_datetime"]]
    source_data1["hour"] = source_data1["data_datetime"].dt.hour
    start_date0, end_date0 = source_data1["data_date"].min(), source_data1["data_date"].max()
    all_dates = pd.date_range(start=start_date0, end=end_date0, freq="d")
    date_data = pd.DataFrame({"data_date": all_dates})

    source_wdata = source_data1.pivot_table(index=["基站id", "data_date"], columns="hour", values="告警label",
                                            aggfunc=np.nansum).reset_index(drop=False)  # 汇总每小时告警编码总和
    # 可能不存在相应的hour
    hour_cols = [i for i in range(24)]
    for hour_col in hour_cols:
        if not hour_col in source_wdata.columns:
            source_wdata.loc[:, hour_col] = np.NaN
    source_wdata = source_wdata[["基站id", "data_date"] + hour_cols]
    if para.ft_n_job == -1:  # 不使用多进程
        all_ft_data = source_wdata.groupby("基站id").apply(
            lambda x: generate_ft_data(x, start_date0, date_data, ft_nday=para.input_len)).reset_index(drop=False).drop(
            "level_1", axis=1)
        all_ft_data.rename(columns={"data_date": "date"}, inplace=True)
        print('使用单进程生成退服数据 in {}s'.format(time.clock() - start))
    # 特征
    # 使用多进程
    else:
        all_es_ids = list(set(source_wdata["基站id"])) # list
#         pd.DataFrame(all_es_ids).to_csv('{}/2222.csv'.format(Org_path), index=False, encoding='gbk')
        grp_id_num = math.ceil(len(all_es_ids) / para.ft_n_job) # 9
        source_wdata_list = []
        for i in range(para.ft_n_job):
            if i < para.ft_n_job - 1:
                p_ids = all_es_ids[i * grp_id_num:(i + 1) * grp_id_num]
            else:
                print(i*grp_id_num)
                p_ids = all_es_ids[i * grp_id_num:]
            p_data = source_wdata.loc[source_wdata["基站id"].isin(p_ids)]
            source_wdata_list.append(p_data)
#         pd.DataFrame(source_wdata_list).to_csv('{}/111111.csv'.format(Org_path), index=False, encoding='gbk')
        def generate_ft_task(pswdata, result_list):
            p_ft_data = pswdata.groupby("基站id").apply(lambda x: generate_ft_data(x, start_date0, date_data, ft_nday=para.input_len)).reset_index(drop=False).drop("level_1", axis=1)
            result_list.append(p_ft_data)

        ft_jobs = []
        ft_results = Manager().list()

        for i in range(len(source_wdata_list)):
            pswd = source_wdata_list[i]
            job = Process(target=generate_ft_task, args=(pswd, ft_results,))
            ft_jobs.append(job)
            job.start()
        for proc in ft_jobs:
            proc.join()

        all_ft_data = pd.concat(ft_results, axis=0)
        all_ft_data.rename(columns={"data_date": "date"}, inplace=True)
        del source_wdata_list, source_wdata
        print('使用{}进程生成退服数据 in {}s'.format(para.ft_n_job, time.clock() - start))

    #     x_cols_1 = [col for col in all_ft_data.columns if not col in ["基站id", "date"]]
    x_cols = [col for col in all_ft_data.columns if not col in ["date"]]

    print("输入特征个数: ", len(x_cols))
    all_ft_data[x_cols] = all_ft_data[x_cols].fillna(-1)  # -1 填充缺失值
    ft_file = "{}/materials/{}_input_features.txt".format(Org_path,para.model_id)

    if para.mode == "predict":
        if os.path.exists(ft_file):
            x_cols0 = load_model_infos(ft_file)  # 获取模型特征列并校验
            #             x_cols0.remove('基站id')
            assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
                                                                  "原模型要求输入：{}。".format(",".join(x_cols),
                                                                                       ",".join(x_cols0)))
            return all_ft_data, x_cols0
        else:
            return all_ft_data, x_cols

    # 获取标签，多进程
    start = time.clock()
    if para.ft_n_job == -1:  # 不使用多进程
        all_label_data = get_pos_neg_dates(para, source_data1, all_dates, real_warns, pred_day=para.target_long)
        print('使用单进程生成退服标签 in {}s'.format(time.clock() - start))

    else:
        all_es_ids = list(set(source_data1["基站id"]))
        grp_id_num = math.ceil(len(all_es_ids) / para.ft_n_job)
        source_data1_list = []
        for i in range(para.ft_n_job):
            if i < para.ft_n_job - 1:
                p_ids = all_es_ids[i * grp_id_num:(i + 1) * grp_id_num]
            else:
                p_ids = all_es_ids[i * grp_id_num:]
            p_data = source_data1.loc[source_data1["基站id"].isin(p_ids)]
            source_data1_list.append(p_data)

        def generate_label_task(psd, result_list):
            p_label_data = get_pos_neg_dates(para, psd, all_dates, real_warns, pred_day=para.target_long)
            result_list.append(p_label_data)

        label_jobs = []
        label_results = Manager().list()
        for i in range(len(source_data1_list)):
            psd1 = source_data1_list[i]
            job = Process(target=generate_label_task, args=(psd1, label_results,))
            label_jobs.append(job)
            job.start()
        for proc in label_jobs:
            proc.join()

        all_label_data = pd.concat(label_results, axis=0)
        del source_data1_list, source_data1, source_data
        print('使用{}进程生成退服标签 in {}s'.format(para.ft_n_job, time.clock() - start))

    all_label_data["true_label"] = all_label_data["true_label"].astype(int)
    all_data = pd.merge(all_label_data, all_ft_data, how="left", on=["基站id", "date"])  # 有label的才能训练和验证
    all_data.dropna(subset=x_cols, how="all", axis=0, inplace=True)  # 排除冗余数据
    ###删除前半年负样本
    #     mid = datetime.datetime(2020, 7, 15)
    #     all_data["date"] = pd.to_datetime(all_data["date"].dt.date)
    #     all_data = all_data[~((all_data['true_label']==0) & (all_data['date']< mid))]
    ###
    all_data[x_cols] = all_data[x_cols].fillna(-1)

    print("输入数据退服告警样本比例:", all_data["true_label"].mean())

    y_col = "true_label"
    Xdata = all_data[x_cols]
    ydata = all_data[y_col]

    if para.mode == 'train':
        save_model_infos(x_cols, ft_file)
        all_data["date"] = pd.to_datetime(all_data["date"], format='%Y-%m-%d %H:%M:%S')
        all_data["week"] = all_data["date"].apply(lambda x: x.isocalendar()[1])  # 增加周标识
        x_cols.append('week')
        Xdata = all_data[x_cols]
        ydata = all_data[y_col]

        Xdata.to_csv(os.path.join(data_dir, "{}_data.csv".format(para.model_id)), index=False)
        ydata.to_csv(os.path.join(data_dir, "{}_label.csv".format(para.model_id)), index=False)

    elif para.mode == "evaluate":
        if os.path.exists(ft_file):
            #         if False:
            x_cols0 = load_model_infos(ft_file)  # 获取模型特征列并校验
            #             x_cols0.del('基站id')
            assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
                                                                  "原模型要求输入：{}。".format(",".join(x_cols),
                                                                                       ",".join(x_cols0)))
            Xdata = Xdata[x_cols0]

    return Xdata, ydata  # gen_train_data, train, evaluate


def predict_run(Org_path,source_data1, model_id, save_path, real_warns,Factory_C,mode_name):
    para.model_id = model_id
    all_warns0 = list(set(source_data1["告警名称"]))
    start_time0, end_time0 = source_data1["data_datetime"].min(), source_data1["data_datetime"].max()

    source_data1["告警开始时间"] = pd.to_datetime(source_data1["告警开始时间"])

    source_data1 = source_data1.sort_values(by=["基站id", "告警开始时间"], ascending=True)  # 先排好序
    source_data1["data_date"] = pd.to_datetime(source_data1["告警开始时间"].dt.date)
    source_data1["data_hour"] = source_data1["告警开始时间"].dt.hour

    source_data1["data_datetime"] = pd.to_datetime(source_data1["data_date"].map(str) + source_data1["data_hour"].map(
        lambda x: ' {}:00:00'.format(str(x) if len(str(x)) > 1 else "0" + str(x))))
    del source_data1["data_hour"]

    # with open('./materials/{}_labelencoder.json'.format(para.model_id), "r") as json_f:
    with open('{}/materials/{}_labelencoder.json'.format(Org_path,para.model_id), "r") as json_f:

        warn_scores = json.load(json_f)
        print('call the existed emb_model')
    for warn_name in all_warns0:  # 可能有些新出现的告警类型
        if not warn_name in warn_scores.keys():
            warn_scores[warn_name] = 0

    start = time.clock()
    para.mode = 'evaluate'
    data_pkg = gen_model_data(para, source_data1, warn_scores, real_warns,Org_path)
    # data_pkg.to_csv('./data_pkg.csv',encoding='gbk',index=False)

    print('gen_model_data in ', time.clock() - start)

    pred_input = data_pkg[0]
    y_label = data_pkg[1]
    pred_output1 = 0
    pred_output2 = 0

    def load_model(para, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("model: {} does not exist in directory: {}.".format(para.model_id, model_path))
        local_model = joblib.load(model_path)
        return local_model

    for i in range(1, 6):
        xgb_model_path = "{}/materials/xgb_model_{}_{}.pkl".format(Org_path,para.model_id, i)
        lgb_model_path = "{}/materials/lgb_model_{}_{}.pkl".format(Org_path,para.model_id, i)
        xgbmd = load_model(para, xgb_model_path)
        lgbmd = load_model(para, lgb_model_path)
        pred_output1 += xgbmd.predict_proba(pred_input, ntree_limit=xgbmd.best_iteration)[:, 1]  # xgb_weight[i-1]*
        pred_output2 += lgbmd.predict(pred_input, num_iteration=lgbmd.best_iteration)
        pred_output_prob = (pred_output1 / 5 + pred_output2 / 5) / 2

    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix

    def evaluate_model_result(true_value, pred_values):
        f1 = f1_score(true_value, pred_values)
        acc = accuracy_score(true_value, pred_values)
        prec = precision_score(true_value, pred_values)
        rec = recall_score(true_value, pred_values)
        cf_mat = confusion_matrix(true_value, pred_values)
        return {"f1:": f1, "acc:": acc, "prec:": prec, "rec:": rec, "confusion:": cf_mat.tolist()}

    pred_output_label = [int(v >= 0.5) for v in pred_output_prob]
    eval_result1 = classification_report(y_label, pred_output_label, output_dict=True)
    eval_result2 = evaluate_model_result(y_label, pred_output_label)
    print("模型验证报告:\n", eval_result1)
    print(eval_result2)

    para.mode = 'predict'
    pred_ft_data_p, ft_cols = gen_model_data(para, source_data1, warn_scores, real_warns,Org_path)
    pred_input = pred_ft_data_p[ft_cols]
    # pred_ft_data_p.to_csv('./pred_ft_data_p-del6.csv',encoding='gbk',index=False)
    # pred_input.to_csv('./pred_input-del6.csv',encoding='gbk',index=False)
    # pred_input = pd.read_csv('./pred_input-del6.csv',encoding='gbk')
    # pred_ft_data_p = pd.read_csv('./pred_ft_data_p-del6.csv',encoding='gbk')

    pred_output1 = 0
    pred_output2 = 0
    for i in range(1, 6):
        xgb_model_path = "{}/materials/xgb_model_{}_{}.pkl".format(Org_path,para.model_id, i)
        lgb_model_path = "{}/materials/lgb_model_{}_{}.pkl".format(Org_path,para.model_id, i)
        xgbmd = load_model(para, xgb_model_path)
        lgbmd = load_model(para, lgb_model_path)
        pred_output1 += xgbmd.predict_proba(pred_input, ntree_limit=xgbmd.best_iteration)[:, 1]
        pred_output2 += lgbmd.predict(pred_input, num_iteration=lgbmd.best_iteration)
    pred_output_prob = (pred_output1 / 5 + pred_output2 / 5) / 2
    pred_output_label = [int(v >= 0.5) for v in pred_output_prob]
    pred_ft_data_p["pred_probability"] = pred_output_prob
    pred_ft_data_p["pred_label"] = pred_output_label

    submit_data = pred_ft_data_p[["基站id", "date", "pred_label", "pred_probability"]]

    # submit_data.to_csv('./submit_data_del6.csv',encoding='gbk',index=False)

    submit_data['date'] = pd.to_datetime(submit_data['date'], format="%Y-%m-%d")
    dates = list(set(submit_data['date']))
    dates.sort(reverse=False)
    proc_list = []
    for i in dates[:-3]:
        print('*' * 20, i, '*' * 20)
        predict_time_0 = i - datetime.timedelta(days=1)
        predict_time_1 = i - datetime.timedelta(days=7)
        TFdate = i
        date = pd.date_range(i, i + datetime.timedelta(days=2))
        date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()

        TF_list = submit_data[submit_data['date'] == i]
        tempall = set()

        date2 = pd.date_range(i - datetime.timedelta(days=7), i - datetime.timedelta(days=1))
        date_list2 = date2.astype(str).map(lambda x: x.replace('-', '')).tolist()
        tempall2 = []
        tempall11 = []
        for k in date_list:
            day = str(k).replace('-', '')
            Alert_date = pd.read_csv(r'{}/Data/故障_处理_{}_delJZ_{}.csv'.format(Org_path,day,mode_name), encoding='gbk', engine='python')
            Alert_date = Alert_date[Alert_date['告警名称'].isin(real_warns)]
            Alert_date['基站id'] = Alert_date['基站id'].astype(int).astype(str)
            Alert_id = set(Alert_date['基站id'])
            tempall = tempall | Alert_id

            Alert_date["data_hour"] = pd.to_datetime(Alert_date['告警开始时间']).dt.hour
            Alert_date = Alert_date.loc[
                ((Alert_date["data_hour"] <= 6) | (Alert_date["data_hour"] >= 23))]  # 退服告警发生时间：23：00-6：00

            Alert_date['基站id'] = Alert_date['基站id'].astype(int).astype(str)
            ntf_id = list(set(Alert_date['基站id']))
            tempall11.extend(ntf_id)

        for m in date_list2:
            day1 = str(m).replace('-', '')
            try:
                nighttf_data = pd.read_csv(r'{}/Data/故障_处理_{}_delJZ_{}.csv'.format(Org_path,day1,mode_name), encoding='gbk')
                nighttf_data = nighttf_data[nighttf_data['告警名称'].isin(real_warns)]
                nighttf_data["data_hour"] = pd.to_datetime(nighttf_data['告警开始时间']).dt.hour
                nighttf_data = nighttf_data.loc[
                    ((nighttf_data["data_hour"] <= 6) | (nighttf_data["data_hour"] >= 23))]  # 退服告警发生时间：23：00-6：00

                nighttf_data['基站id'] = nighttf_data['基站id'].astype(int).astype(str)
                nighttf_id = list(set(nighttf_data['基站id']))
                tempall2.extend(nighttf_id)
            except:
                print('{}/Data/故障_处理_{}_delJZ_{}.csv is not existed!'.format(Org_path,day1,mode_name))

        from collections import Counter
        c = dict(Counter(list(tempall2)))
        a = list(c.keys())
        b = list(c.values())
        c = {'基站id': a,
             '前七天夜间退服天数': b}
        tf_days = pd.DataFrame(c)

        c1 = dict(Counter(list(tempall11)))
        a1 = list(c1.keys())
        b1 = list(c1.values())
        c1 = {'基站id': a1,
              '后三天夜间退服天数': b1}
        tf_days1 = pd.DataFrame(c1)

        TF_list['基站id'] = TF_list['基站id'].astype(int).astype(str)
        TF_list['Label_5'] = TF_list['基站id'].map(lambda x: 1 if x in tempall else 0)
        TF_list = pd.merge(TF_list, tf_days, how='left')
        TF_list = pd.merge(TF_list, tf_days1, how='left')

        TF_list.sort_values(by='pred_probability', ascending=False, inplace=True)
        proc_list.append(TF_list)

    #     TF_list1 = TF_list[TF_list['前七天夜间退服天数']<6]
    #     print('实际退服基站数量：%s,top5_7D预测正确数量：%s(不包含连续夜间退服>6)'%(len(tempall),sum(TF_list1['Label_5'].to_list()[:5])))
    #     TF_list1.sort_values(by='pred_probability_6D',ascending=False,inplace=True)
    #     print('实际退服基站数量：%s,top5_6D预测正确数量：%s(不包含连续夜间退服>6)'%(len(tempall),sum(TF_list1['Label_5'].to_list()[:5])))
    proc_res = pd.concat(proc_list)
    # proc_res1 = proc_res[proc_res['pred_probability'] > 0.5]
    # print('precesion:',sum(proc_res1['Label_5'].to_list())/proc_res1.shape[0])
    # proc_res2 = proc_res[(proc_res['pred_probability'] > 0.5)&(proc_res['前七天夜间退服天数']<6)]
    # print('precesion(不包含连续夜间退服>6):',sum(proc_res2['Label_5'].to_list())/proc_res2.shape[0])
    proc_res.to_csv('{}/{}'.format(Org_path,save_path), index=False, encoding='gbk')
    x = proc_res
    print(classification_report(x['Label_5'],x['pred_label']))

def run_predict_Pre(Org_path, Factory_C, City_name, mode_name, date_low_train, date_high_train, date_low_pre,
                    date_high_pre, real_warns):
    # Org_path = 'D:/Pycharm/LN-4G-ZX/SY_4G_ZX/'
    # date_low_train='20210301'
    # date_high_train='20210731'
    # date_low_pre='20210725'
    # date_high_pre='20210816'
    source_data1 = pd.read_csv('{}/{}-{}_source_data_del6D.csv'.format(Org_path, date_low_pre, date_high_pre),
                               encoding='gbk')
    model_id = "{}_{}-{}_{}_{}_NoList_Del6D".format(City_name, date_low_train, date_high_train, mode_name, Factory_C)
    save_path = '{}_{}_{}_Res_NoList_delGC6D.csv'.format(City_name, mode_name, Factory_C)
    predict_run(Org_path,source_data1, model_id, save_path, real_warns,Factory_C,mode_name)
    source_data1 = pd.read_csv('{}/{}-{}_source_data_del7D.csv'.format(Org_path, date_low_pre, date_high_pre),
                               encoding='gbk')
    model_id = "{}_{}-{}_{}_{}_NoList_Del7D".format(City_name, date_low_train, date_high_train, mode_name, Factory_C)
    save_path = '{}_{}_{}_Res_NoList_delGC7D.csv'.format(City_name, mode_name, Factory_C)
    predict_run(Org_path,source_data1, model_id, save_path, real_warns,Factory_C,mode_name)

# for i in dates[:-1]:
#     predict_time_0 = i-datetime.timedelta(days=1)  
#     predict_time_1 = i-datetime.timedelta(days=7)  

#     TFdate = i

#     date = pd.date_range(i,i+datetime.timedelta(days=2))
#     date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()
#     TF_list = submit_data[submit_data['date']==i]
#     TF_list1 = TF_list[TF_list['pred_probability']>=0.66]

#     tempall = []
#     for k in [0]:
#         distr = str(i).split()[0].replace('-','')
#         Alert_date = pd.read_csv('./data/origin-test/告警日志{}.csv'.format(distr),encoding='gbk',engine='python')
# #         Alert_date = pd.read_csv('./predict-0104-0117/故障_{}_delJZ.csv'.format(distr),encoding='gbk',engine='python')
#         tempall.append(Alert_date)

#     Alert_all = pd.concat(tempall,axis=0)
#     Alert_select = Alert_all[(Alert_all['告警名称'] == '射频单元维护链路异常告警') |
#                                         (Alert_all['告警名称'] == '小区不可用告警')
#                                         | (Alert_all['告警名称'] == '网元连接中断')| (Alert_all['告警名称'] == 'eNodeB退服告警')
#         | (Alert_all['告警名称'] == '传输光接口异常')
#     ]
#     TF_count = Alert_select.groupby('基站id')['告警名称'].count()
#     TF_sum = len(TF_count)
#     TF_count_D = TF_count.to_frame()
#     TF_count_D.reset_index(inplace=True)

#     TF_merge = pd.merge(TF_list1,TF_count,on='基站id',how='left')

#     TF_num = len(TF_merge)
#     TF_merge_FN = TF_merge.fillna(-1)
#     TF_1_count = TF_merge_FN['告警名称'].value_counts()
#     TF_1_count_D = TF_1_count.to_frame()
#     try:
#         False_num = TF_1_count_D.loc[-1,'告警名称']
#     except:
#         False_num = 0
#     precision = 1-(False_num/TF_num)

#     print('Result_{}_{}_{}_{}'.format(TFdate,TF_sum,TF_num,str(precision)[:6]))
