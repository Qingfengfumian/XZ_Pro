import numpy as np
import pandas as pd
import os
import re
import datetime
import csv
import json
import csv
from utils import mkdir
from utils import real_warns_5G,real_warns_4G
from utils import mysql_dealdata
csv.field_size_limit(500 * 1024 * 1024)

def validation(para):
    distr_list = para.distr_list
    ftype = para.ftype
    date = para.date
    yy1 = int(date.split('-')[1][0:4])
    mm1 = int(date.split('-')[1][4:6])
    dd1 = int(date.split('-')[1][6:8])
    base_date = datetime.date(yy1, mm1, dd1)
    Inspect_date1 = datetime.datetime.strftime(base_date + datetime.timedelta(days=1), "%Y%m%d")
    Inspect_date = datetime.datetime.strftime(base_date - datetime.timedelta(days=2), "%Y%m%d")
    predict_time_0 = datetime.datetime.strptime(Inspect_date1, "%Y%m%d")
    date_low_Time = predict_time_0 - datetime.timedelta(days=3)
    date_high_Time = predict_time_0 - datetime.timedelta(days=1)
    date = pd.date_range(str(date_low_Time).split(' ')[0].replace('-', ''),
                         str(date_high_Time).split(' ')[0].replace('-', ''))
    date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()
    # 新建文件夹
    mkdir(r"{}/OutService_{}".format(para.out_path, Inspect_date1))

    for distr in distr_list:
        print(distr + '退服验证')
        distr_dict = {'拉萨':'LS','昌都':'CD','山南':'SN','林芝':'LZ','日喀则':'RKZ','那曲':'NQ','阿里':'AL'}
        distr_new = distr_dict.get(distr)
        try:
            TF_list = pd.read_csv('{}/OutService_{}/OutService_{}_{}.csv'.format(para.out_path,Inspect_date, Inspect_date, distr_new), encoding='gbk')
        except:
            print('{}无{}退服预测数据'.format(distr,Inspect_date))
            continue
        TF_list1 = TF_list.copy()
        # TF_list1 = TF_list[TF_list['pred_probability']>=0.6]
        # TF_list1 = TF_list[TF_list['pred_label']==1]
        tempall = []
        for date_1 in date_list:
            alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(distr, date_1), encoding='gbk')
            tempall.append(alert_data_part)

        Alert_all = pd.concat(tempall, axis=0)

        # --------- 20211021 修改匹配退服告警清单 -----------
        TF_list = real_warns_5G + real_warns_4G
        TF_list_Pd = pd.DataFrame(TF_list, columns=['告警名称']).drop_duplicates()  # 剔除+去重
        Alert_select = pd.merge(Alert_all, TF_list_Pd)

        TF_count = Alert_select.groupby('基站id')['告警名称'].count()
        TF_sum = len(TF_count)
        TF_count_D = TF_count.to_frame()
        TF_count_D.reset_index(inplace=True)

        TF_merge = pd.merge(TF_list1, TF_count, on='基站id', how='left')

        TF_num = len(TF_merge)
        TF_merge_FN = TF_merge.fillna(-1)
        TF_1_count = TF_merge_FN['告警名称'].value_counts()
        TF_1_count_D = TF_1_count.to_frame()
        try:
            False_num = TF_1_count_D.loc[-1, '告警名称']
        except:
            False_num = 0
        precision = 1 - (False_num / TF_num)

        TF_merge.rename(columns={'告警名称': '实际退服告警数量'}, inplace=True)
        TF_merge['实际退服告警数量'].fillna(0,inplace=True)
        # TF_merge.to_csv('{}/OutServiceTest_{}_{}.csv'.format(para.out_path, Inspect_date, distr_new), encoding='gbk',index=False)
        TF_merge.to_csv('{}/OutService_{}/OutServiceTest_{}_{}.csv'.format(para.out_path, Inspect_date1, Inspect_date, distr_new), encoding='gbk',index=False)
        print("退服验证清单生成完成，结果保存在: {}。".format('{}/OutServiceTest_{}_{}.csv'.format(para.out_path, Inspect_date, distr_new)))
