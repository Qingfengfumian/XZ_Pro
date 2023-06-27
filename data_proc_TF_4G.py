import pandas as pd
import numpy as np
import os
import random
import json
import math
import time
from multiprocessing import Process, Manager

from utils import load_model_infos,find_new_file,save_model_infos,mkdir
from feature_engineer import generate_ft_data, get_pos_neg_dates, cut_warn_series2
from model import build_emb_model
def load_data(para,mode,distr):
    ftype = para.ftype
    if mode=='train_TF':
        date = para.train_date1
    if mode=='predict':
        date = para.date
    data_tmp = pd.read_csv(
        './Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_delJZ.csv'.format(distr, mode,ftype, date),
        encoding='gbk', engine='python')  # index_col=0)
    if "制式" in data_tmp.columns:
        data_tmp.drop("制式", axis=1, inplace=True)
        # if "告警持续时间" in data_tmp.columns:
        #     data_tmp.rename(columns={"告警持续时间": "告警结束时间"}, inplace=True)
    source_data = data_tmp

    source_data = source_data.drop_duplicates(subset=["基站id", "告警名称", "告警开始时间"], keep='first')  # 去除重复告警

    source_data["告警开始时间"] = pd.to_datetime(source_data["告警开始时间"])
    source_data = source_data.sort_values(by=["基站id", "告警开始时间"], ascending=True)  # 先排好序
    source_data["data_date"] = pd.to_datetime(source_data["告警开始时间"].dt.date)
    source_data["data_hour"] = source_data["告警开始时间"].dt.hour
    source_data["data_datetime"] = pd.to_datetime(source_data["data_date"].map(str) + source_data["data_hour"].map(
        lambda x: ' {}:00:00'.format(str(x) if len(str(x)) > 1 else "0" + str(x))))

    # source_data = source_data.loc[~((source_data["基站id"] == 498624) & (source_data["告警名称"].isin(real_warns))
    #                                 & (source_data["data_hour"] == 2))]  # 删除节电基站记录
    source_data.drop("data_hour", axis=1, inplace=True)

    all_warns0 = list(set(source_data["告警名称"]))

    start_time0, end_time0 = source_data["data_datetime"].min(), source_data["data_datetime"].max()
    return source_data, all_warns0, start_time0, end_time0
def gen_emb_series(para, source_data):
    # 使用第二种方式生成告警序列
    all_warn_series = []
    for grp_data in source_data.groupby("基站id"):
        gdata = grp_data[1]
        id = grp_data[0]
        result = cut_warn_series2(gdata, back_day=para.input_len, front_day=para.target_long,
                                  drop_hour_duplicate=para.drop_hour_duplicate)
        all_warn_series.append(result)

    flattern_warn_series = []
    for warn_s in all_warn_series:
        for _, wls in warn_s.items():
            flattern_warn_series.extend(wls)
    flattern_warn_series = [wl for wl in flattern_warn_series if len(wl) > 0]
    random.shuffle(flattern_warn_series)  # 应该乱序重置?

    return flattern_warn_series

def gen_model_data(para,distr,source_data, warn_scores, input_len=7, target_long=3):
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
    if para.ft_n_job == -1: #不使用多进程
        all_ft_data = source_wdata.groupby("基站id").apply(
            lambda x: generate_ft_data(x, start_date0, date_data, ft_nday=para.input_len)).reset_index(drop=False).drop(
            "level_1", axis=1)
        all_ft_data.rename(columns={"data_date": "date"}, inplace=True)
        print('使用单进程生成退服数据 in {}s'.format(time.clock()-start))
    # 特征
    # 使用多进程
    else:
        all_es_ids = list(set(source_wdata["基站id"]))
        grp_id_num = math.ceil(len(all_es_ids) / para.ft_n_job)
        source_wdata_list = []
        for i in range(para.ft_n_job):
            if i < para.ft_n_job - 1:
                p_ids = all_es_ids[i * grp_id_num:(i + 1) * grp_id_num]
            else:
                p_ids = all_es_ids[i * grp_id_num:]
            p_data = source_wdata.loc[source_wdata["基站id"].isin(p_ids)]
            source_wdata_list.append(p_data)

        def generate_ft_task(pswdata, result_list):
            p_ft_data = pswdata.groupby("基站id").apply(
                lambda x: generate_ft_data(x, start_date0, date_data, ft_nday=input_len)).reset_index(drop=False).drop(
                "level_1", axis=1)
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
        print('使用{}进程生成退服数据 in {}s'.format(para.ft_n_job,time.clock()-start))

    x_cols = [col for col in all_ft_data.columns if not col in ["基站id", "date"]]
    print("输入特征个数: ", len(x_cols))
    all_ft_data[x_cols] = all_ft_data[x_cols].fillna(-1)  # -1 填充缺失值

    if para.mode == "predict":
        ft_file = find_new_file(r"./Data/{}/Inter_Result/{}_特征标题/".format(distr, para.ftype))
        if os.path.exists(ft_file):
            x_cols0 = load_model_infos(ft_file)  # 获取模型特征列并校验
            assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
                                                                  "原模型要求输入：{}。".format(",".join(x_cols),
                                                                                       ",".join(x_cols0)))
            return all_ft_data, x_cols0
        else:
            return all_ft_data, x_cols

        if para.mode == "predict":
            if os.path.exists(ft_file):
                x_cols0 = load_model_infos(ft_file)  # 获取模型特征列并校验
                assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
                        "原模型要求输入：{}。".format(",".join(x_cols), ",".join(x_cols0)))
                return all_ft_data, x_cols0
            else:
                return all_ft_data, x_cols

    # 获取标签，多进程
    start = time.clock()
    if para.ft_n_job == -1: #不使用多进程
        all_label_data = get_pos_neg_dates(para, source_data1, all_dates, pred_day=para.target_long)
        print('使用单进程生成退服标签 in {}s'.format(time.clock()-start))

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
            p_label_data = get_pos_neg_dates(para,psd, all_dates, pred_day=target_long)
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
        print('使用{}进程生成退服标签 in {}s'.format(para.ft_n_job,time.clock()-start))

    all_label_data["true_label"] = all_label_data["true_label"].astype(int)
    all_data = pd.merge(all_label_data, all_ft_data, how="left", on=["基站id", "date"])  # 有label的才能训练和验证
    all_data.dropna(subset=x_cols, how="all", axis=0, inplace=True)  # 排除冗余数据

    all_data[x_cols] = all_data[x_cols].fillna(-1)

    print("输入数据退服告警样本比例:", all_data["true_label"].mean())

    y_col = "true_label"
    Xdata = all_data[x_cols]
    ydata = all_data[y_col]

    if para.mode == 'train_TF':
        ft_file = "./Data/{}/Inter_Result/{}_特征标题/{}_input_features.txt".format(distr, para.ftype,
                                                                                para.train_date1)
        save_model_infos(x_cols, ft_file)
        all_data["date"] = pd.to_datetime(all_data["date"], format='%Y-%m-%d %H:%M:%S')
        all_data["week"] = all_data["date"].apply(lambda x: x.isocalendar()[1])  # 增加周标识
        x_cols.append('week')
        Xdata = all_data[x_cols]
        ydata = all_data[y_col]

        Xdata.to_csv(r"./Data/{}/Alert_Samp/Samp_{}/退服_样本_Xdata_{}.csv".format(distr, para.ftype, para.train_date1),
                     index=False)
        ydata.to_csv(r"./Data/{}/Alert_Samp/Samp_{}/退服_样本_ydata_{}.csv".format(distr, para.ftype, para.train_date1),
                     index=False)

    elif para.mode == "evaluate":
        ft_file = find_new_file(
            r"./Data/{}/Inter_Result/4G/{}_特征标题/".format(distr, para.ftype))
        if os.path.exists(ft_file):
            x_cols0 = load_model_infos(ft_file) # 获取模型特征列并校验
            assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
                                        "原模型要求输入：{}。".format(",".join(x_cols), ",".join(x_cols0)))
            Xdata = Xdata[x_cols0]

    return Xdata, ydata  # gen_train_data, train, evaluate


def proc_TF_4G(para,distr):
    print(distr)
    mode = para.mode
    source_data, all_warns0, start_time0, end_time0 = load_data(para,mode,distr)
    Inter_path = ["./Data/{}/Inter_Result/4G/{}_特征标题/".format(distr,para.ftype),
               './Data/{}/Inter_Result/4G/{}_退服_模型/'.format(distr, para.ftype),
               './Data/{}/Inter_Result/4G/{}_特征编码'.format(distr, para.ftype)
                  ]
    for file_path in Inter_path:
        if os.path.exists(file_path)==False:
            mkdir(file_path)
    if mode=='train_TF':
        flat_warn_series = gen_emb_series(para, source_data)
        emb_warn_scores = build_emb_model(para, flat_warn_series, all_warns0)
        gen_model_data(para, source_data, emb_warn_scores,distr)
    if (mode=='predict') | (mode == "evaluate"):
        labelcode_path = find_new_file("./Data/{}/Inter_Result/4G/{}_特征编码/".format(distr,para.ftype))
        if not os.path.exists(labelcode_path):
            raise FileNotFoundError("no new pre-trained embedding scores found in directory: {}.".format(
                './Data/{}/Inter_Result/{}_特征编码/'))
        with open(labelcode_path, "r") as json_f:
            warn_scores = json.load(json_f)
        for warn_name in all_warns0:  # 可能有些新出现的告警类型
            if not warn_name in warn_scores.keys():
                warn_scores[warn_name] = 0
        if para.mode == "predict":
            # assert para.result_path != "" and para.result_path != " ", "please specify a file to save forecast output."
            pred_ft_data, ft_cols = gen_model_data(para,distr, source_data, warn_scores)  # 检查是否所有告警类型都出现过
            return pred_ft_data, ft_cols
        elif para.mode == "evaluate":
            data_pkg = gen_model_data(para, distr,source_data, warn_scores)  # 检查是否所有告警类型都出现过
            return data_pkg

