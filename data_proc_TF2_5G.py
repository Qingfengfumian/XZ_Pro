import pandas as pd
import numpy as np
import os
import random
import json
import math
import time
from multiprocessing import Process, Manager

import argparse
import os
import multiprocessing
import time
import pandas as pd

import numpy as np

import math

import random
from multiprocessing import Process, Manager

from numpy.lib.stride_tricks import as_strided as stride
from tqdm import tqdm

from utils import real_warns_5G,load_model_infos,find_new_file2,save_model_infos,mkdir
from feature_engineer import generate_ft_data, get_pos_neg_dates, cut_warn_series2
from model import build_emb_model
def load_data(para,mode,distr,day=7):
    del_days = day
    ftype = para.ftype
    if mode=='train_TF':
        date = para.train_date1
    if mode=='predict':
        date = para.date
    data_tmp = pd.read_csv(
        './Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_delJZ_5G.csv'.format(distr, mode,ftype, date),
        encoding='gbk', engine='python')  # index_col=0)
    if "制式" in data_tmp.columns:
        data_tmp.drop("制式", axis=1, inplace=True)
        # if "告警持续时间" in data_tmp.columns:
        #     data_tmp.rename(columns={"告警持续时间": "告警结束时间"}, inplace=True)
    source_data1 = data_tmp
    ###---------------------
    ### 按分钟去重
    source_data1['data_minute'] = source_data1['告警开始时间'].map(lambda x:str(x)[:-3])
    source_data1 = source_data1.drop_duplicates(subset=["基站id", "告警名称", "data_minute"], keep='first')  # 去除重复告警
    del source_data1['data_minute']

    source_data1["告警开始时间"] = pd.to_datetime(source_data1["告警开始时间"])

    source_data1 = source_data1.sort_values(by=["基站id", "告警开始时间"], ascending=True)  # 先排好序
    source_data1["data_date"] = pd.to_datetime(source_data1["告警开始时间"].dt.date)
    source_data1["data_hour"] = source_data1["告警开始时间"].dt.hour

    source_data1["data_datetime"] = pd.to_datetime(source_data1["data_date"].map(str) + source_data1["data_hour"].map(
        lambda x: ' {}:00:00'.format(str(x) if len(str(x)) > 1 else "0" + str(x))))
    # del source_data1["data_hour"]

    start_time0, end_time0 = source_data1["data_datetime"].min(), source_data1["data_datetime"].max()
    source_data = source_data1.copy()
    def roll(df: pd.DataFrame, window: int, **kwargs):
        v = df.reset_index().values
        dim0, dim1 = v.shape
        stride0, stride1 = v.strides
        stride_values = stride(v, (dim0 - (window - 1), window, dim1), (stride0, stride0, stride1))
        rolled_df = pd.concat({
            row: pd.DataFrame(values[:, 1:], columns=df.columns, index=values[:, 0].flatten())
            for row, values in zip(df.index[window - 1:], stride_values)
        })
        return rolled_df.groupby(level=0, **kwargs)

    def trandform_grp_data(grp_sd):
        # grp_sd = grp_sd.sort_values(by="告警开始时间", ascending=True)
        min_time = grp_sd["data_datetime"].min()  # - pd.to_timedelta("1h")
        max_time = grp_sd["data_datetime"].max() + pd.to_timedelta("1h")
        date_data = pd.DataFrame({"data_datetime": pd.date_range(min_time, max_time, freq="h")})
        grp_sd = pd.merge(date_data, grp_sd, how="left", on="data_datetime")
        grp_sd["data_hour"] = grp_sd["data_datetime"].dt.hour  # 补充缺失值
        grp_sd["data_date"] = grp_sd["data_datetime"].dt.date  # 补充缺失值
        grp_sd2 = grp_sd.loc[
            ((grp_sd["data_hour"] <= 6) | (grp_sd["data_hour"] >= 23)) & (
                grp_sd["告警名称"].isin(real_warns_5G))]  # 小时数不完整，空的自动不计入  # 23-6
        #         grp_sd2 = grp_sd.loc[
        #             (grp_sd["告警名称"].isin(real_warns_5G))]  # 全天
        del grp_sd
        return grp_sd2

    def delete_electricity_warn(roll_sd, es_id):
        over_rwh_count = pd.DataFrame(columns=["基站id", "data_date", "data_hour"])

        # 寻找相邻小时
        all_hours = list(roll_sd.columns)

        def hour_add(v1):
            return v1 + 1 if v1 != 23 else 0

        def hour_minus(v1):
            return v1 - 1 if v1 != 0 else 23

        valid_hours0 = []
        for hour in all_hours:
            roll_hours = [hour]
            pre_hour, aft_hour = hour_minus(hour), hour_add(hour)
            if pre_hour in all_hours:
                roll_hours.append(pre_hour)
            if aft_hour in all_hours:
                roll_hours.append(aft_hour)
            if sum(roll_sd[roll_hours].max(axis=1)) >= del_days:  # 连续7天出现告警
                valid_hours0.extend(roll_hours)

        valid_hours = list(set(valid_hours0))
        for valid_hour in valid_hours:
            relate_dates = list(roll_sd.loc[roll_sd[valid_hour] > 0].index)
            over_rwh_count = over_rwh_count.append(
                pd.DataFrame({"基站id": [es_id] * len(relate_dates), "data_date": relate_dates,
                              "data_hour": [valid_hour] * len(relate_dates)}), ignore_index=True)
        return over_rwh_count

    def get_elect_datetimes(fmd):
        id_datetimes = []
        for index, row in fmd.iterrows():
            dates = row["data_date"]
            hour = row["data_hour"]
            id_datetimes.extend([pd.to_datetime(date) + pd.to_timedelta("{}h".format(str(hour))) for date in dates])
        id_datetimes = list(set(id_datetimes))
        return id_datetimes

    # filter_metadata_li = {}
    filter_data_list = []
    # for grp in tqdm(source_data.groupby("基站id")):
    for grp in source_data.groupby("基站id"):
        es_id = grp[0]
        grp_data = grp[1]
        grp_sd = trandform_grp_data(grp_data)
        if grp_sd.shape[0] > 0:
            real_min_date, real_max_date = grp_sd["data_date"].min(), grp_sd["data_date"].max()
            if (real_max_date - real_min_date).days >= 7:  # 按7天滚动
                # grp_sd = grp_sd.sort_values(by="告警开始时间", ascending=True)
                grp_sd["has_real_warn"] = (~grp_sd["告警名称"].isna()).astype(int)
                grp_wsd = grp_sd.pivot_table(index="data_date", columns="data_hour", values="has_real_warn",
                                             aggfunc="mean")  # .reset_index(drop=False)
                real_dates = pd.date_range(start=real_min_date, end=real_max_date)
                grp_wsd = grp_wsd.reindex(real_dates, fill_value=0).sort_index(ascending=True)  # 原数据已乱序
                grp_wsd = grp_wsd.fillna(0)
                elec_info_d = roll(grp_wsd, window=7).apply(delete_electricity_warn, es_id)
                if elec_info_d.shape[0] > 0:
                    id_filer_times = get_elect_datetimes(elec_info_d)
                    # filter_metadata_li[es_id] = id_filer_times
                    grp_data = grp_data.loc[~(grp_data["data_datetime"].isin(id_filer_times))]
        filter_data_list.append(grp_data)

    del source_data
    source_data = pd.concat(filter_data_list, axis=0)
    source_data = source_data.sort_values(by=["基站id", "告警开始时间"], ascending=True)  # 排序
    del source_data["data_hour"]
    all_warns0 = list(set(source_data["告警名称"]))

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

def gen_model_data(para,day,distr,source_data, warn_scores, input_len=7, target_long=3):
    start = time.clock()
    source_data["告警label"] = source_data["告警名称"].apply(lambda x: warn_scores[x])
    source_data1 = source_data[["告警label", "告警名称", "告警开始时间", "基站id", "data_date", "data_datetime"]]
    source_data1["hour"] = source_data1["data_datetime"].dt.hour
    start_date0, end_date0 = source_data1["data_date"].min(), source_data1["data_date"].max()
    all_dates = pd.date_range(start=start_date0, end=end_date0, freq="d")
    date_data = pd.DataFrame({"data_date": all_dates})

    source_wdata = source_data1.pivot_table(index=["基站id", "data_date"], columns="hour", values="告警label",
                                            aggfunc=np.nansum).reset_index(drop=False)  # 汇总每小时告警编码总和
    #可能不存在相应的hour
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

    # x_cols = [col for col in all_ft_data.columns if not col in ["基站id", "date"]]
    x_cols = [col for col in all_ft_data.columns if not col in ["date"]]
    print("输入特征个数: ", len(x_cols))
    all_ft_data[x_cols] = all_ft_data[x_cols].fillna(-1)  # -1 填充缺失值

    if para.mode == "predict":
        fea_title_path = "./Data/{}/Inter_Result/5G/{}_特征标题/".format(distr, para.ftype)
        ft_file = find_new_file2(fea_title_path,day)
        if os.path.exists(os.path.join(fea_title_path,ft_file)):
            x_cols0 = load_model_infos(os.path.join(fea_title_path,ft_file))  # 获取模型特征列并校验
            assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
                                                                  "原模型要求输入：{}。".format(",".join(x_cols),
                                                                                       ",".join(x_cols0)))
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

    print("%s天剔除，输入数据退服告警样本比例:"%day, all_data["true_label"].mean())

    y_col = "true_label"
    Xdata = all_data[x_cols]
    ydata = all_data[y_col]

    if para.mode == 'train_TF':
        ft_file = "./Data/{}/Inter_Result/5G/{}_特征标题/input_features_del{}_{}.txt".format(distr,para.ftype,day,
                                                                                para.train_date1)
        save_model_infos(x_cols, ft_file)
        all_data["date"] = pd.to_datetime(all_data["date"], format='%Y-%m-%d %H:%M:%S')
        all_data["week"] = all_data["date"].apply(lambda x: x.isocalendar()[1])  # 增加周标识
        x_cols.append('week')
        Xdata = all_data[x_cols]
        ydata = all_data[y_col]

        Xdata.to_csv(r"./Data/{}/Alert_Samp/Samp_{}/退服_样本_Xdata_del{}_{}_5G.csv".format(distr, para.ftype, day, para.train_date1),
                     index=False)
        ydata.to_csv(r"./Data/{}/Alert_Samp/Samp_{}/退服_样本_ydata_del{}_{}_5G.csv".format(distr, para.ftype, day, para.train_date1),
                     index=False)

    elif para.mode == "evaluate":
        fea_title_path = "./Data/{}/Inter_Result/5G/{}_特征标题/".format(distr, para.ftype)
        ft_file = find_new_file2(fea_title_path,day)
        if os.path.exists(os.path.join(fea_title_path, ft_file)):
            x_cols0 = load_model_infos(os.path.join(fea_title_path, ft_file))  # 获取模型特征列并校验
            assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
                                                                  "原模型要求输入：{}。".format(",".join(x_cols),
                                                                                       ",".join(x_cols0)))
            Xdata = Xdata[x_cols0]
        return Xdata, ydata  # gen_train_data, train, evaluate


def proc_TF_5G(para,distr,day):
    mode = para.mode
    source_data, all_warns0, start_time0, end_time0 = load_data(para,mode,distr,day)
    Inter_path = ["./Data/{}/Inter_Result/5G/{}_特征标题/".format(distr,para.ftype),
               './Data/{}/Inter_Result/5G/{}_退服_模型/'.format(distr, para.ftype),
               './Data/{}/Inter_Result/5G/{}_特征编码'.format(distr, para.ftype)
                  ]
    for file_path in Inter_path:
        if os.path.exists(file_path)==False:
            mkdir(file_path)
    if mode=='train_TF':
        flat_warn_series = gen_emb_series(para, source_data)
        emb_warn_scores = build_emb_model(para, day,flat_warn_series, all_warns0)
        gen_model_data(para, day,source_data, emb_warn_scores,distr)
    if (mode=='predict') | (mode == "evaluate"):
        fea_cod_path = "./Data/{}/Inter_Result/5G/{}_特征编码/".format(distr,para.ftype)
        labelcode_path = find_new_file2(fea_cod_path,day)
        if not os.path.exists(os.path.join(fea_cod_path,labelcode_path)):
            raise FileNotFoundError("no new pre-trained embedding scores found in directory: {}.".format(
                fea_cod_path))
        with open(os.path.join(fea_cod_path,labelcode_path), "r") as json_f:
            warn_scores = json.load(json_f)
        for warn_name in all_warns0:  # 可能有些新出现的告警类型
            if not warn_name in warn_scores.keys():
                warn_scores[warn_name] = 0
        if para.mode == "predict":
            # assert para.result_path != "" and para.result_path != " ", "please specify a file to save forecast output."
            pred_ft_data, ft_cols = gen_model_data(para,day,distr, source_data, warn_scores)  # 检查是否所有告警类型都出现过
            return pred_ft_data, ft_cols
        elif para.mode == "evaluate":
            data_pkg = gen_model_data(para, day,distr,source_data, warn_scores)  # 检查是否所有告警类型都出现过
            return data_pkg

if __name__=="__main__":
    pass
