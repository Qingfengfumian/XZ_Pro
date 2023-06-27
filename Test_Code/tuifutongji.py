import pandas as pd
from pandas.testing import assert_frame_equal
import os
import csv
import json
import pickle
import datetime

# ---------------- 本地退服告警统计 ------------------------
path = 'C:/Users/x1carbon\Desktop\LN_服务器数据\退服/'
data_list = os.listdir(path)
c =[]
for i in data_list:
    if os.path.isdir(path+i):
        tuifu_data = os.listdir(path+i)
        for j in tuifu_data:
            if 'OutServiceTest_' in j:
                print(j)
                a = path+i+'/'+j
                try:b = pd.read_csv(a,encoding='gbk')
                except:b = pd.read_excel(a)
                c.append(b)
e = pd.concat(c,axis=0)
# e.sort_values(by='eventTime', inplace=True)
e.to_csv('{}/退服汇总0510-0516.csv'.format(path),encoding='gbk',index=False)


# # ---------------- 服务器退服告警统计 ------------------------
# for district in ['宝鸡','咸阳','西安','商洛','安康', '渭南', '延安', '榆林','铜川','汉中']:#
#     date_low = '20210809'
#     date_high = '20210820'
#     date = pd.date_range(date_low, date_high)
#     date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()
#     TF_sum_sum = []
#     for date_loop in date_list:
#         predict_time_low = datetime.datetime.strptime(date_loop,'%Y%m%d')-datetime.timedelta(days=7)
#         predict_time_high = datetime.datetime.strptime(date_loop,'%Y%m%d')-datetime.timedelta(days=1)
#         predict_time = str(predict_time_low).split(' ')[0].replace('-','')+'-'+str(predict_time_high).split(' ')[0].replace('-','')
#
#         date_low_Time = datetime.datetime.strptime(date_loop,'%Y%m%d')
#         date_high_Time = datetime.datetime.strptime(date_loop,'%Y%m%d') + datetime.timedelta(days=2)
#         date11 = pd.date_range(str(date_low_Time).split(' ')[0].replace('-', ''),
#                              str(date_high_Time).split(' ')[0].replace('-', ''))
#         date_list_yanz = date11.astype(str).map(lambda x: x.replace('-', '')).tolist()
#         print('./Data/{}/Inspect_List/华为/TFPre_{}_5G.csv'.format(district, predict_time))
#         TF_list = pd.read_csv('./Data/{}/Inspect_List/华为/TFPre_{}_5G.csv'.format(district, predict_time), encoding='gbk')
#         # except:
#         #     continue
#         TF_list1 = TF_list.copy()
#         # TF_list1 = TF_list[TF_list['pred_probability']>=0.6]
#         # TF_list1 = TF_list[TF_list['pred_label']==1]
#         tempall = []
#         for date_1 in date_list_yanz:
#             alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(district, date_1), encoding='gbk')
#             tempall.append(alert_data_part)
#
#         Alert_all = pd.concat(tempall, axis=0)
#         # Alert_select = Alert_all[(Alert_all['告警名称'] == '射频单元维护链路异常告警')
#         #                          | (Alert_all['告警名称'] == '小区不可用告警')
#         #                          | (Alert_all['告警名称'] == '网元连接中断') | (Alert_all['告警名称'] == 'eNodeB退服告警') | (
#         #                                  Alert_all['告警名称'] == '传输光接口异常告警')]
#         Alert_select = Alert_all[(Alert_all['告警名称'] == '射频单元维护链路异常告警')
#                                  | (Alert_all['告警名称'] == '小区不可用告警') | (Alert_all['告警名称'] == 'eNodeB退服告警')
#                                  | (Alert_all['告警名称'] == '网元连接中断') | (Alert_all['告警名称'] == 'gNodeB退服告警') | (
#                                          Alert_all['告警名称'] == 'NR小区不可用告警')]
#         TF_count = Alert_select.groupby('基站id')['告警名称'].count()
#         TF_sum = len(TF_count)
#         TF_count_D = TF_count.to_frame()
#         TF_count_D.reset_index(inplace=True)
#
#         TF_merge = pd.merge(TF_list1, TF_count, on='基站id', how='left')
#
#         # TF_num = len(TF_merge)
#         # TF_merge_FN = TF_merge.fillna(-1)
#         # TF_1_count = TF_merge_FN['告警名称'].value_counts()
#         # TF_1_count_D = TF_1_count.to_frame()
#         # try:
#         #     False_num = TF_1_count_D.loc[-1, '告警名称']
#         # except:
#         #     False_num = 0
#         # precision = 1 - (False_num / TF_num)
#
#         TF_merge.rename(columns={'告警名称': '实际退服告警数量'}, inplace=True)
#         TF_merge['实际退服告警数量'].fillna(0, inplace=True)
#         TF_sum_sum.append(TF_merge)
#     fff = pd.concat(TF_sum_sum, axis=0)
#     # fff.drop_duplicates(subset=['网元名称'],inplace=True)
#     fff.to_csv('./Data/{}_退服统计.csv'.format(district), encoding='gbk',index=False)