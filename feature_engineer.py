# -*- coding:utf-8 -*-

import pandas as pd, numpy as np
import datetime
from utils import real_warns_4G

# 划分告警序列，生成embedding的"句子"
def cut_warn_series(grp_data, interval=24, least_num=1, drop_edge=False):
    if len(grp_data) < 2:  # 至少2，即首尾不重合
        if least_num == 1 and not drop_edge:
            return [grp_data['告警名称'].tolist()], [[np.NaN]]
        else:
            return [], []
    grp_data = grp_data.sort_values(by=["告警开始时间"], ascending=True)
    grp_data["intervals"] = grp_data["告警开始时间"].diff() / np.timedelta64(3600, 's')
    grp_data["cut_label"] = (grp_data["intervals"] > interval).astype(int)
    warn_series = []
    interval_series = []
    warns = grp_data["告警名称"].tolist()
    intervals = grp_data["intervals"].tolist()
    cut_labels = grp_data["cut_label"].tolist()
    range_num = len(grp_data)
    start = 0
    if cut_labels[1] == 1:
        warn_series.append(warns[0])
        interval_series.append(intervals[0])
        start = 1
    for i in range(1, range_num):
        if i == range_num - 1:
            if cut_labels[i] == 1:
                warn_series.append(warns[i])
                interval_series.append(intervals[i])
            else:
                warn_series.append(warns[start:])
                interval_series.append(intervals[start:])
        else:
            if cut_labels[i] == 1:
                warn_series.append(warns[start:i])
                interval_series.append(intervals[start:i])
                start = i

    if drop_edge:
        if len(warn_series) < 3:
            warn_series, interval_series = [], []
        else:
            warn_series = warn_series[1:len(warn_series) - 1]
            interval_series = interval_series[1:len(interval_series) - 1]

    if least_num >= 1:
        warn_series = [ww for ww in warn_series if len(ww) >= least_num]
        interval_series = [iv for iv in interval_series if len(iv) >= least_num]

    return warn_series, interval_series


def cut_warn_series2(gdata, back_day=7, front_day=3, drop_hour_duplicate=False):
    gdata = gdata.sort_values(by="告警开始时间", ascending=True)
    real_gdata = gdata.loc[gdata["告警名称"].isin(real_warns_4G)]
    time_col = "告警开始时间"
    if drop_hour_duplicate:
        real_gdata = real_gdata.drop_duplicates(subset=["data_datetime", "告警名称"], keep="first")
        time_col = "data_datetime"
    real_warn_round_series = {real_warn: [] for real_warn in real_warns_4G}
    for i in range(real_gdata.shape[0]):
        row = real_gdata.iloc[i]
        warn_time = row[time_col]
        start_cal_time = warn_time - datetime.timedelta(days=back_day)
        end_cal_time = warn_time + datetime.timedelta(days=front_day)
        real_warn = row["告警名称"]
        spec_warn_list = gdata.loc[
            (gdata[time_col] <= end_cal_time) & (gdata[time_col] >= start_cal_time), "告警名称"].tolist()  # 需按时间顺序排好
        real_warn_round_series[real_warn].append(spec_warn_list)
    return real_warn_round_series


# 生成特征数据
def generate_ft_data(gwdata, start_date0, date_data, ft_nday=7):
    start_pred_date = start_date0 + datetime.timedelta(days=ft_nday)
    gawdata = pd.merge(date_data, gwdata, on="data_date", how="left")
    gawdata.sort_values(by="data_date", ascending=True, inplace=True)
    ft_cols = [i for i in range(24)]
    gftdata = gawdata[["data_date"] + ft_cols]
    gftdata["data_date"] = gftdata["data_date"] + datetime.timedelta(days=1)  # 锚定预测日期第一天
    gftdata.rename(columns={i: "date-1_{}".format(str(i)) for i in ft_cols}, inplace=True)
    if ft_nday > 1:
        date1_cols = ["date-1_{}".format(str(i)) for i in ft_cols]
        for i in range(2, ft_nday + 1):
            gftdata[["date-{}_{}".format(str(i), str(ii)) for ii in ft_cols]] = gftdata[date1_cols].shift(i-1)
    gftdata = gftdata.loc[gftdata["data_date"] >= start_pred_date]
    gftdata.dropna(axis=0, subset=gftdata.columns[1:], how="all", inplace=True)  # 去除冗余数据
    return gftdata
# 生成日期对应的标签
def get_pos_neg_dates(para,sdata, all_dates, pred_day=3):
    true_label_data = pd.DataFrame(columns=["基站id", "date", "true_label"])
    for grp in sdata.groupby("基站id"):
        df = grp[1]
        df_pos = df[df["告警名称"].isin(real_warns_4G)]
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
        if para.mode in ["predict","evaluate"]:
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