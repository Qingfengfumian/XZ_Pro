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
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# real_warns = ["MME衍生基站退服","LTE小区退出服务","MME衍生小区退服","基站退服","网元断链告警"]
# model_dir = './materials'
# data_dir = './materials'
# material_dir = './materials'
# save_path = './materials'


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
    parser.add_argument('--xgb_show_loss', type=bool, default=False)
    parser.add_argument('--drop_hour_duplicate', type=bool, default=False)  # embedding 模型参数
    parser.add_argument('--emb_size', type=int, default=50)
    parser.add_argument('--emd_exist', type=bool, default=False)
    parser.add_argument('--emb_window', type=int, default=7)
    parser.add_argument('--emb_iter', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.50)  # 分类阈值
    parser.add_argument('--data_path', type=str,
                        default="./data/XianYang/train-0701-1231")  # HanZhong/TongChuan 输入数据文件路径
    parser.add_argument('--data_exist', type=bool, default=False)  # 是否直接使用已生成数据集
    parser.add_argument('--log_path', type=str, default="./logs/model4JZ@HZ0120-0103ALL.txt")  # log日志保存路径
    parser.add_argument('--log_write', type=str, default="a")  # log日志保存路径
    parser.add_argument('--result_path', type=str,
                        default="./model_result/predict_model4JZ@XY0701-1130.csv")  # 预测结果保存路径
    parser.add_argument('--mode', type=str, default="train")  # 运行模式: "train", "train_eval", "evaluate", "predict"
    parser.add_argument('--test_size', type=float, default=0.20)  # 测试数据集比例
    parser.add_argument('--model_id', type=str, default="AK_20200701-20210628_NoList_Del7D_3TF")  # 模型唯一标识
    parser.add_argument('--ft_n_job', type=int, default=multiprocessing.cpu_count())  # 特征工程采用多少个线程
    #     para = parser.parse_args()
    para = parser.parse_args(args=[])
    return para


para = params_setup()


def cut_warn_series2(gdata, real_warns, back_day=7, front_day=3, drop_hour_duplicate=False):
    gdata = gdata.sort_values(by="告警开始时间", ascending=True)
    real_gdata = gdata.loc[gdata["告警名称"].isin(real_warns)]
    time_col = "告警开始时间"
    if drop_hour_duplicate:
        real_gdata = real_gdata.drop_duplicates(subset=["data_datetime", "告警名称"], keep="first")
        time_col = "data_datetime"  # h级别  data_date d级别
    real_warn_round_series = {real_warn: [] for real_warn in real_warns}
    for i in range(real_gdata.shape[0]):
        row = real_gdata.iloc[i]
        warn_time = row[time_col]
        start_cal_time = warn_time - datetime.timedelta(days=back_day)
        end_cal_time = warn_time  # + datetime.timedelta(days=front_day)
        real_warn = row["告警名称"]
        spec_warn_list = gdata.loc[
            (gdata[time_col] <= end_cal_time) & (gdata[time_col] >= start_cal_time), "告警名称"].tolist()  # 需按时间顺序排好
        real_warn_round_series[real_warn].append(spec_warn_list)
    return real_warn_round_series


##########################只取前7天告警#########################################
def cut_warn_series3(gdata, real_warns, back_day=7, front_day=3, drop_hour_duplicate=False):
    gdata = gdata.sort_values(by="告警开始时间", ascending=True)
    real_gdata = gdata.loc[gdata["告警名称"].isin(real_warns)]
    time_col = "告警开始时间"
    #     if drop_hour_duplicate:
    real_gdata = real_gdata.drop_duplicates(subset=["data_date", "告警名称"], keep="first")
    time_col = "data_date"  # h级别  data_date d级别
    real_warn_round_series = {real_warn: [] for real_warn in real_warns}
    for i in range(real_gdata.shape[0]):
        row = real_gdata.iloc[i]
        warn_time = row[time_col]
        start_cal_time = warn_time - datetime.timedelta(days=back_day)
        end_cal_time = warn_time  # + datetime.timedelta(days=front_day)
        real_warn = row["告警名称"]
        spec_warn_list = gdata.loc[
            (gdata[time_col] < end_cal_time) & (gdata[time_col] >= start_cal_time), "告警名称"].tolist()  # 需按时间顺序排好
        real_warn_round_series[real_warn].append(spec_warn_list)
    return real_warn_round_series


##########################################################################

def gen_emb_series(para, source_data, real_warns):
    # 使用第二种方式生成告警序列

    all_warn_series = []
    for grp_data in source_data.groupby("基站id"):
        gdata = grp_data[1]
        id = grp_data[0]
        result = cut_warn_series2(gdata, real_warns, back_day=para.input_len, front_day=para.target_long,
                                  drop_hour_duplicate=para.drop_hour_duplicate)
        all_warn_series.append(result)

    flattern_warn_series = []
    for warn_s in all_warn_series:
        for _, wls in warn_s.items():
            flattern_warn_series.extend(wls)
    flattern_warn_series = [wl for wl in flattern_warn_series if len(wl) > 0]
    random.shuffle(flattern_warn_series)  # 应该乱序重置?

    return flattern_warn_series


def build_emb_model(para, warn_series, all_warns0, real_warns,Org_path):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    wv_model = word2vec.Word2Vec(warn_series, size=para.emb_size, window=para.emb_window, min_count=1,
                                 iter=para.emb_iter)
    #     emb_path = './'
    #     if os.path.exists(emb_path):
    #         mkdir(emb_path)
    #     wv_model.save("./embedding_model.pkl")
    # 计算每种告警类型的得分
    warn_scores = {}
    for warn_name in all_warns0:
        sim_score = 0
        if warn_name in wv_model.wv.vocab.keys():
            for real_warn in real_warns:
                sim_score += wv_model.wv.similarity(real_warn, warn_name)
        else:
            print(warn_name)
        warn_scores[warn_name] = sim_score if sim_score > 0 else 0  # 最小值设为0
    ##############3
    #     print(warn_scores)

    max_score = max(warn_scores.values())
    warn_scores2 = copy.deepcopy(warn_scores)

    # 重置退服告警的编码
    for real_warn in real_warns:
        warn_scores2[real_warn] = 2 * max_score
    # 保存编码结果
    #     save_path = r"./{}_特征编码/".format(para.distr,para.ftype)
    #     if os.path.exists(save_path)==False:
    #         mkdir(save_path)
    with open("{}/materials/{}_labelencoder.json".format(Org_path,para.model_id),
              "w") as json_f:  # 保存到数据目录（不同模型可能共用数据）
        json.dump(warn_scores2, json_f)
    return warn_scores2


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
                lambda x: generate_ft_data(x, start_date0, date_data, ft_nday=para.input_len)).reset_index(
                drop=False).drop(
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
        print('使用{}进程生成退服数据 in {}s'.format(para.ft_n_job, time.clock() - start))
    ##############################
    #     x_cols = [col for col in all_ft_data.columns if not col in ["基站id", "date"]]
    x_cols = [col for col in all_ft_data.columns if not col in ["date"]]
    x_cols_fea = [col for col in all_ft_data.columns if not col in ["date", '基站id']]

    print("输入特征个数: ", len(x_cols))
    #     all_ft_data[x_cols] = all_ft_data[x_cols].fillna(-1)  # -1 填充缺失值
    ft_file = "{}_input_features.txt".format(para.model_id)

    if para.mode == "predict":
        print('2021.12.27已注释')
        # if os.path.exists(ft_file):
        #     x_cols0 = load_model_infos(ft_file)  # 获取模型特征列并校验
        #     assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
        #                                                           "原模型要求输入：{}。".format(",".join(x_cols),
        #                                                                                ",".join(x_cols0)))
        #     return all_ft_data, x_cols0
        # else:
        #     return all_ft_data, x_cols

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
    all_data.dropna(subset=x_cols_fea, how="all", axis=0, inplace=True)  # 排除冗余数据

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

    ## 增加跨越年份的判断
    def get_week(y):
        if y.isocalendar()[0] == 2021:
            week = y.isocalendar()[1] + 53
        else:
            week = y.isocalendar()[1]
        return week

    if para.mode == 'train':
        save_model_infos(x_cols, ft_file,Org_path)
        all_data["date"] = pd.to_datetime(all_data["date"], format='%Y-%m-%d %H:%M:%S')
        all_data["week"] = all_data["date"].apply(lambda x: get_week(x))  # 增加周标识
        x_cols.append('week')
        #         x_cols.append('date')
        Xdata = all_data[x_cols]
        ydata = all_data[y_col]


    #         Xdata.to_csv(os.path.join(data_dir, "{}_data_.csv".format(para.model_id)), index=False)
    #         ydata.to_csv(os.path.join(data_dir, "{}_label.csv".format(para.model_id)), index=False)

    elif para.mode == "evaluate":
        print('2021.12.27已注释')
        # if os.path.exists(ft_file):
        #     x_cols0 = load_model_infos(ft_file) # 获取模型特征列并校验
        #     assert all([col in x_cols for col in x_cols0]), print("特征数据列与原模型不一致, 特征列：{}, "
        #                                 "原模型要求输入：{}。".format(",".join(x_cols), ",".join(x_cols0)))
        #     Xdata = Xdata[x_cols0]

    return Xdata, ydata  # gen_train_data, train, evaluate


def build_xgb_model(para, Xdata, ydata,Org_path, eval_data=None):
    xgbm = XGBClassifier(max_depth=6,  # xy tc 5
                         learning_rate=0.1,
                         n_estimators=1000,
                         objective="binary:logistic",
                         nthread=multiprocessing.cpu_count(), gamma=0, min_child_weight=1,
                         max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                         base_score=0.5, seed=0, missing=None)
    folds = GroupKFold(n_splits=5)
    groups = Xdata['week']
    xgbms = []
    x_cols = [col for col in Xdata.columns if not col in ["week"]]
    Xdata = Xdata[x_cols]
    for n_fold, (train_index, test_index) in enumerate(folds.split(Xdata, ydata, groups), start=1):
        print('the {}-th cross-validation train'.format(n_fold))
        X_train = Xdata.iloc[train_index]
        y_train = ydata.iloc[train_index]
        evali = [(Xdata.iloc[test_index], ydata.iloc[test_index])]
        sub_xgbm = xgbm.fit(X_train, y_train, eval_set=evali, eval_metric=['logloss', 'auc', 'error'],
                            early_stopping_rounds=100, verbose=True)
        if para.xgb_show_loss == True:
            results = sub_xgbm.evals_result()
            epochs = len(results['validation_0']['error'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = pyplot.subplots()
            ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
            # ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
            ax.legend()
            pyplot.ylabel('Log Loss')
            pyplot.title('XGBoost Log Loss')
            pyplot.show()
            # plot classification error
            fig, ax = pyplot.subplots()
            ax.plot(x_axis, results['validation_0']['error'], label='Train')
            # ax.plot(x_axis, results['validation_1']['error'], label='Test')
            ax.legend()
            pyplot.ylabel('Classification Error')
            pyplot.title('XGBoost Classification Error')
            pyplot.show()
        model_path = os.path.join("{}/materials/xgb_model_{}_{}.pkl".format(Org_path,para.model_id, n_fold))
        joblib.dump(sub_xgbm, model_path)  # 保存
        xgbms.append(sub_xgbm)
    print("xgboost 模型训练完毕, 已更新在: {}。".format(model_path))
    return xgbms


def build_lgb_model(para, Xdata, ydata,Org_path):
    
    param = {
        #         'num_leaves': 50,
        'objective': 'binary',
        'max_depth': 6,
        'learning_rate': 0.1,
        'max_bin': 100,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        'min_data_in_leaf': 100,
        'verbose': -1,
        'num_threads': multiprocessing.cpu_count()
    }
    param['metric'] = ['auc', 'binary_logloss']
    folds = GroupKFold(n_splits=5)
    # 采用5折交叉验证
    groups = Xdata['week']
    lgbms = []
    x_cols = [col for col in Xdata.columns if not col in ["week"]]
    Xdata = Xdata[x_cols]
    for n_fold, (train_index, test_index) in enumerate(folds.split(Xdata, ydata, groups), start=1):
        lgb_train_data = lgb.Dataset(Xdata.iloc[train_index], ydata.iloc[train_index])
        lgb_valid_data = lgb.Dataset(Xdata.iloc[test_index], ydata.iloc[test_index])
        lgbm_1 = lgb.train(params=param, train_set=lgb_train_data, valid_sets=lgb_valid_data, num_boost_round=1000,
                           early_stopping_rounds=100, verbose_eval=1
                           )
        model_path = "{}/materials/lgb_model_{}_{}.pkl".format(Org_path,para.model_id, n_fold)
        joblib.dump(lgbm_1, model_path)  # 保存
        lgbms.append(lgbm_1)
    print("lightgbm 模型训练完毕, 已更新在: {}。".format(model_path))
    return lgbms


def save_model_infos(list_info, file_name,Org_path):
    with open("{}/materials/{}".format(Org_path,file_name), "w") as f:
        for item in list_info:
            f.write(item + "\n")

        # source_data11 = pd.read_csv('./data/20200701-20210228_source_data_del7D.csv',encoding='gbk')


# source_data22 = pd.read_csv('./data/20210429-20210628_source_data_del7D.csv',encoding='gbk')


def train_run(source_data1, model_id, real_warns,Org_path):
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

    # date_s = pd.to_datetime('2021-01-01',format='%Y-%m-%d %H:%M:%S')
    # all_data['告警开始时间'] = pd.to_datetime(all_data['告警开始时间'],format='%Y-%m-%d')
    # test_data = all_data[all_data['告警开始时间']>=date_s]
    # source_data = all_data[all_data['告警开始时间']<date_s]

    if para.data_exist == True:  # 调用原有数据
        print('call the existed dataset')
        Xdata = pd.read_csv("{}/materials/{}_data.csv".format(Org_path,para.model_id))
        ydata = pd.read_csv("{}/materials/{}_label.csv".format(Org_path,para.model_id), header=None)
        data_pkg = (Xdata, ydata)
    else:  # 生成新emb模型和数据
        if para.emd_exist == True:
            with open('{}/materials/{}_labelencoder.json'.format(Org_path,para.model_id), "r") as json_f:
                emb_warn_scores = json.load(json_f)
                print('call the existed emb_model')
            start = time.clock()
            data_pkg = gen_model_data(para, source_data1, emb_warn_scores, real_warns,Org_path)
            print('gen_model_data in ', time.clock() - start)
        else:
            start = time.clock()
            flat_warn_series = gen_emb_series(para, source_data1, real_warns)
            print('gen_emb_series in ', time.clock() - start)
            start = time.clock()
            emb_warn_scores = build_emb_model(para, flat_warn_series, all_warns0, real_warns,Org_path)
            print('build_emb_model in ', time.clock() - start)
            start = time.clock()
            data_pkg = gen_model_data(para, source_data1, emb_warn_scores, real_warns,Org_path)
            print('gen_model_data in ', time.clock() - start)

    start = time.clock()
    lgbmd = build_lgb_model(para, data_pkg[0], data_pkg[1],Org_path)
    print('build_lgb_model in ', time.clock() - start)
    start = time.clock()
    xgbmd = build_xgb_model(para, data_pkg[0], data_pkg[1],Org_path)
    print('build_xgb_model in ', time.clock() - start)



def run_train_TR(Org_path, Factory_C, City_name, mode_name, date_low_train, date_high_train, real_warns):
    # Org_path = 'D:/Pycharm/LN-4G-ZX/SY_4G_ZX/'
    print('开始执行6D')
    source_data1 = pd.read_csv('{}/{}-{}_source_data_del6D.csv'.format(Org_path, date_low_train, date_high_train),
                               encoding='gbk')
    model_id = "{}_{}-{}_{}_{}_NoList_Del6D".format(City_name, date_low_train, date_high_train, mode_name, Factory_C)
    train_run(source_data1, model_id, real_warns,Org_path)

    print('开始执行7D')
    source_data1 = pd.read_csv('{}/{}-{}_source_data_del7D.csv'.format(Org_path, date_low_train, date_high_train),
                               encoding='gbk')
    model_id = "{}_{}-{}_{}_{}_NoList_Del7D".format(City_name, date_low_train, date_high_train, mode_name, Factory_C)
    train_run(source_data1, model_id, real_warns,Org_path)
