import argparse
import os
import multiprocessing
import time
import pandas as pd
# from xgboost import XGBClassifier
# import lightgbm as lgb
# from gensim.models import word2vec
# from sklearn.model_selection import train_test_split,GroupKFold
import numpy as np
# from matplotlib import pyplot
# import joblib
import logging
import copy
# import json
import math
import multiprocessing
import datetime
import random
from multiprocessing import Process, Manager
from collections import deque
from numpy.lib.stride_tricks import as_strided as stride
from tqdm import tqdm

# from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix

# real_warns = ["eNodeB退服告警","基站退服","MME衍生基站退服","小区不可用告警","网元连接中断"] 
# real_warns = ["MME衍生基站退服","LTE小区退出服务","MME衍生小区退服","基站退服","网元断链告警"]

ft_n_job = 32


def read_source_data1(data_path, alarm_conti, real_warns):
    #     source_data_list = []
    #     for data_file in os.listdir(data_path):
    #         if data_file.endswith(".csv"):
    #             try:
    #                 data_tmp = pd.read_csv(os.path.join(data_path, data_file), encoding="utf-8",engine='python')
    #             except:
    #                 data_tmp = pd.read_csv(os.path.join(data_path, data_file), encoding="gbk", engine='python')
    #             if "制式" in data_tmp.columns:
    #                 data_tmp.drop("制式", axis=1, inplace=True)
    #             if "告警持续时间" in data_tmp.columns:
    #                 data_tmp.rename(columns={"告警持续时间": "告警结束时间"},inplace=True)
    #             source_data_list.append(data_tmp)
    #     source_data = pd.concat(source_data_list, axis=0)

    try:
        data_tmp = pd.read_csv(data_path, low_memory=False)
    except:
        data_tmp = pd.read_csv(data_path, encoding="gbk", low_memory=False)
    if "制式" in data_tmp.columns:
        data_tmp.drop("制式", axis=1, inplace=True)
    #     if "告警持续时间" in data_tmp.columns:
    #         data_tmp.rename(columns={"告警持续时间": "告警结束时间"},inplace=True)

    source_data = data_tmp

    source_data = source_data.drop_duplicates(subset=["基站id", "告警名称", "告警开始时间"], keep='first')  # 去除重复告警
    ###---------------------
    ### 按分钟去重
    source_data['data_minute'] = source_data['告警开始时间'].map(lambda x: str(x)[:-3])
    source_data = source_data.drop_duplicates(subset=["基站id", "告警名称", "data_minute"], keep='first')  # 去除重复告警
    del source_data['data_minute']
    print('去重后数据：', source_data.shape)
    ###---------------------
    source_data["告警开始时间"] = pd.to_datetime(source_data["告警开始时间"])

    source_data = source_data.sort_values(by=["基站id", "告警开始时间"], ascending=True)  # 先排好序
    source_data["data_date"] = pd.to_datetime(source_data["告警开始时间"].dt.date)
    source_data["data_hour"] = source_data["告警开始时间"].dt.hour

    source_data["data_datetime"] = pd.to_datetime(source_data["data_date"].map(str) + source_data["data_hour"].map(
        lambda x: ' {}:00:00'.format(str(x) if len(str(x)) > 1 else "0" + str(x))))

    source_data = source_data.drop_duplicates(subset=["基站id", "告警名称", "告警开始时间"], keep='first')  # 去除重复告警

    # all_warns0 = list(set(source_data["告警名称"]))
    start_time0, end_time0 = source_data["data_datetime"].min(), source_data["data_datetime"].max()
    print('delete data...')

    # 按规则去除节电退服记录：对每个基站滚动看一周(7天)内23点-6点同一小时发生退服的天数，超过3次，就把发生了退服的该时刻告警全部删掉。
    def roll(df: pd.DataFrame, window: int, **kwargs):
        # move index to values
        v = df.reset_index().values

        dim0, dim1 = v.shape
        stride0, stride1 = v.strides

        stride_values = stride(v, (dim0 - (window - 1), window, dim1), (stride0, stride0, stride1))

        rolled_df = pd.concat({
            row: pd.DataFrame(values[:, 1:], columns=df.columns, index=values[:, 0].flatten())
            for row, values in zip(df.index[window - 1:], stride_values)
        })

        return rolled_df.groupby(level=0, **kwargs)

    def trandform_grp_data(grp_sd, real_warns):
        # grp_sd = grp_sd.sort_values(by="告警开始时间", ascending=True)
        min_time = grp_sd["data_datetime"].min()  # - pd.to_timedelta("1h")
        max_time = grp_sd["data_datetime"].max() + pd.to_timedelta("1h")
        date_data = pd.DataFrame({"data_datetime": pd.date_range(min_time, max_time, freq="h")})
        grp_sd = pd.merge(date_data, grp_sd, how="left", on="data_datetime")
        grp_sd["data_hour"] = grp_sd["data_datetime"].dt.hour  # 补充缺失值
        grp_sd["data_date"] = grp_sd["data_datetime"].dt.date  # 补充缺失值
        grp_sd2 = grp_sd.loc[
            ((grp_sd["data_hour"] <= 6) | (grp_sd["data_hour"] >= 23)) & (
                grp_sd["告警名称"].isin(real_warns))]  # 小时数不完整，空的自动不计入  # 23-6
        #         grp_sd2 = grp_sd.loc[
        #             (grp_sd["告警名称"].isin(real_warns))]  # 全天
        del grp_sd
        return grp_sd2

    def delete_electricity_warn(roll_sd, es_id):
        # rwh_count = roll_sd[["data_date", "data_hour"]].groupby(["data_hour"])["data_date"].unique().reset_index(
        #     drop=False)
        # date_lis = rwh_count["data_date"].values
        # date_counts = [dates.shape[0] for dates in date_lis]
        # rwh_count["date_count"] = date_counts
        # over_rwh_count = rwh_count.loc[rwh_count["date_count"] >= 3]
        # over_rwh_count["基站id"] = es_id
        # return over_rwh_count[["基站id", "data_date", "data_hour"]]

        over_rwh_count = pd.DataFrame(columns=["基站id", "data_date", "data_hour"])

        # valid_hours0 = set(roll_sd.loc[:, roll_sd.sum(axis=0) >= 3].columns)
        # if 23 in roll_sd.columns and 0 in roll_sd.columns:
        #     if (roll_sd[23].sum() + roll_sd[0].sum()) >= 3:
        #         valid_hours0.add(23)
        #         valid_hours0.add(0)

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
            if np.sum(roll_sd[roll_hours].values) >= alarm_conti:  # 连续7天出现告警
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

    def select_data(x, real_warns):
        print("len(x)={}".format(len(x)));
        filter_data_list = []
        for grp in tqdm(x.groupby("基站id")):
            es_id = grp[0];
            grp_data = grp[1];
            grp_sd = trandform_grp_data(grp_data, real_warns)
            # if grp_sd["data_date"].nunique() >= 7:  # 按7天滚动
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
        fin_x = pd.concat(filter_data_list, axis=0);
        return fin_x;

    def generate_ft_task(pswdata, result_list, real_warns):
        p_ft_data = select_data(pswdata, real_warns);
        result_list.append(p_ft_data);

    start = time.clock();
    # 按多进程拆分数据
    ft_n_job = 20
    all_es_ids = list(set(source_data["基站id"]))
    grp_id_num = math.ceil(len(all_es_ids) / ft_n_job)
    source_wdata_list = []
    for i in range(ft_n_job):
        if i < ft_n_job - 1:
            p_ids = all_es_ids[i * grp_id_num:(i + 1) * grp_id_num]
        else:
            p_ids = all_es_ids[i * grp_id_num:]
        # 按基站id获取数据
        p_data = source_data.loc[source_data["基站id"].isin(p_ids)]
        source_wdata_list.append(p_data)

    ft_jobs = []
    # 存放进程返回的结果
    ft_results = Manager().list()
    for i in range(len(source_wdata_list)):
        pswd = source_wdata_list[i]
        job = Process(target=generate_ft_task, args=(pswd, ft_results, real_warns,))
        ft_jobs.append(job)
        job.start()
        print('start Process[%s]...' % i);
    for proc in ft_jobs:
        proc.join();

    print("ft_results={}".format(ft_results));
    all_ft_data = pd.concat(ft_results, axis=0)
    all_ft_data.rename(columns={"data_date": "date"}, inplace=True)
    del source_wdata_list, source_data
    print('使用{}进程生成数据 in {}s'.format(ft_n_job, time.clock() - start))

    all_ft_data = all_ft_data.sort_values(by=["基站id", "告警开始时间"], ascending=True)  # 排序
    del all_ft_data["data_hour"]
    all_warns0 = list(set(all_ft_data["告警名称"]))

    ###同小时同类告警 删除
    #     source_data0 = source_data.copy()
    #     source_data0['同小时告警次数'] = source_data0.groupby(by=['基站id','data_datetime','告警名称'])['厂家'].transform('count')
    #     source_data_del = source_data0[source_data0['同小时告警次数']>5]
    #     source_data_del = source_data_del.groupby(by=['基站id','data_datetime','告警名称']).apply(lambda df:df[:5]).reset_index(drop=True)
    #     source_data_no = source_data0[source_data0['同小时告警次数']<=5]
    #     source_data_fin = pd.concat([source_data_del,source_data_no])
    #     source_data = source_data_fin.drop('同小时告警次数',axis=1,inplace=False)

    return all_ft_data, all_warns0, start_time0, end_time0


def source_data_run(Org_path, Factory_C, date_low_train, date_high_train, date_low_pre, date_high_pre, real_warns):
    # Org_path = 'D:/Pycharm/LN-4G-ZX/SY_4G_ZX/'
    # date_low='20210301'
    # date_high='20210731'
    # date_low = '20210725'
    # date_high = '20210816'
    alarm_conti = 6
    source_data1, all_warns1, start_time1, end_time1 = read_source_data1(
        '{}/{}-{}_NoList_ALL.csv'.format(Org_path, date_low_train, date_high_train), alarm_conti, real_warns)
    source_data1.to_csv('{}/{}-{}_source_data_del7D.csv'.format(Org_path, date_low_train, date_high_train),
                        encoding='gbk', index=False)
    alarm_conti = 7
    source_data1, all_warns1, start_time1, end_time1 = read_source_data1(
        '{}/{}-{}_NoList_ALL.csv'.format(Org_path, date_low_train, date_high_train), alarm_conti, real_warns)
    source_data1.to_csv('{}/{}-{}_source_data_del6D.csv'.format(Org_path, date_low_train, date_high_train),
                        encoding='gbk', index=False)

    alarm_conti = 6
    source_data1, all_warns1, start_time1, end_time1 = read_source_data1(
        '{}/{}-{}_NoList_ALL.csv'.format(Org_path, date_low_pre, date_high_pre), alarm_conti, real_warns)
    source_data1.to_csv('{}/{}-{}_source_data_del7D.csv'.format(Org_path, date_low_pre, date_high_pre), encoding='gbk',
                        index=False)
    alarm_conti = 7
    source_data1, all_warns1, start_time1, end_time1 = read_source_data1(
        '{}/{}-{}_NoList_ALL.csv'.format(Org_path, date_low_pre, date_high_pre), alarm_conti, real_warns)
    source_data1.to_csv('{}/{}-{}_source_data_del6D.csv'.format(Org_path, date_low_pre, date_high_pre), encoding='gbk',
                        index=False)
