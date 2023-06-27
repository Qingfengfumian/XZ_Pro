import time,os
import pandas as pd
import numpy as np
from datetime import datetime

import os
import shutil
import json

path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据调研/20230308数据/'
filename = 'ALARM-2023031415-activeAlarm'

A = pd.read_csv(path + '/{}.csv'.format(filename),encoding='utf-8',sep='^')
A.to_csv(path+'/{}-new.csv'.format(filename),encoding='utf-8',index=False)

# A = pd.read_csv(path+'/{}.csv'.format(filename),encoding='gbk',sep='\$\$',header=None)
import csv
s=""
n =[]
with open(path + '/{}.csv'.format(filename),encoding='gbk') as f:
    csvread = csv.reader(f)
    IF_star_deal = 0
    for rows in csvread:
        if rows.__len__()> 0:
            for row in rows:
                s=s+row
            # if "AlarmStatus" in row:
            if "GNodeB" in row:
                IF_star_deal = 1
                n.append(s)
                s = ""
            if IF_star_deal == 1:
                n.append(s)
                s = ""
            else:
                s = ""

        if rows.__len__()== 0:
            n.append(s)
            s=""
        # if rows==['\n']:
        #     n.append(s)
        #     s=""
save_fil=[]
for item in n:
    if item != '':
        save_fil.append(item.split("$$"))
    try:

        print(item.split("$$").__len__())
    except:
        pass

alert_PD = pd.DataFrame(save_fil)

# alert_PD.iloc[0,0]='SpecialtyLv1'
alert_PD.columns = alert_PD.iloc[0,:].values.tolist()
alert_dropna = alert_PD.drop([0], axis=0)
alert_dropna.to_csv(path+'/{}-new.csv'.format(filename),encoding='utf-8',index=False)



# A= pd.read_csv('D:\WeChat1\WeChat Files\wxid_5915759156212\FileStorage\File/2022-09/20220829AIOPS_ICOS_NR.csv',sep='|')
# print(A.shape)
# DD= pd.read_csv('C:/Users/x1carbon\Desktop\无线运维工作台技术规范\安徽融合部署/mod-11546299-85876849_DW_DM_ZY_GNODEB_20220704000000.csv',sep='\$\$',encoding='utf-8')
# DD.dropna(axis=0,inplace=True)
#
# B=A[~A['fault_description'].str.contains('黑点')]
# E=B[~B['fault_description'].str.contains('新疆')]
# print(E.shape)
# CC = DD[0:2000]
# # CC['NODEB_ID'] = CC['NODEB_ID'].map(lambda x:x.split('-')[1])
# CC.reset_index(inplace=True,drop=True)
# E['eci'] = CC['NODEB_ID']
# E['def_cellname_chinese'] = CC['USERLABEL']
# E.to_csv('D:\WeChat1\WeChat Files\wxid_5915759156212\FileStorage\File/2022-09/20220829AIOPS_ICOS_NR-new.csv',index=False,encoding='utf-8',sep='|')
# # ------------ 网元ID解析 ---------------
# path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据调研\样例数据-告警数据/'
# alarm_data = pd.read_csv(path+'/历史测试导出新(1).csv',encoding='gbk')
# alarm_sele = alarm_data[(alarm_data['SPECIALTYEX2']=='4G')| (alarm_data['SPECIALTYEX2']=='5G')]
# alert_data = alarm_sele[['NENAME', 'SPECIALTYEX2', 'EVENT_TIME', 'VENDOR', 'CANCEL_TIME', 'ALARMTITLE']]
#
# nm_date = pd.read_csv('./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv',encoding='gbk')
#
# nm_ID = nm_date[['基站名称','ENODEB_ID']]
# merge_data1 = pd.merge(alert_data,nm_ID,how='left',left_on='NENAME',right_on='基站名称')
# merge_data1.drop(columns=['基站名称'],inplace=True)
#
# nm_AREA = nm_date[['基站名称','所属地市']]
# merge_data2 = pd.merge(merge_data1,nm_AREA,how='left',left_on='NENAME',right_on='基站名称')
# merge_data2.drop(columns=['基站名称'],inplace=True)
#
# alarm_sele = merge_data2.dropna()
#
# alarm_sele.to_csv(path + '/告警new.csv',encoding='utf-8',index=False)
# A = {"addInfo":"基站制式:L-N; 影响制式:N; 部署标识:AUTODID_20210325_743d010a-8cba-11eb-8000-fa163eb13; 累计时长(s):90; 描述信息:2177751; gNodeBId:2178123;deployment:SA_NSA","alarmId":"140789150","alarmSeq":95384294,"alarmStatus":1,"alarmTitle":"gNodeB Xn接口故障告警","alarmType":"信令系统","eventTime":"2022-05-01 00:08:06","locationInfo":"gNodeB名称:LS_城关区_加荣路琅塞金贝贝双语幼儿园_TT_CPGS_H5H; 具体问题:建立失败产生","neName":"LS_城关区_加荣路琅塞金贝贝双语幼儿园_TT_CPGS_H5H","neType":"GNB","neUID":"5401HWWX1MNE3221291436","objectName":"LS_城关区_加荣路琅塞金贝贝双语幼儿园_TT_CPGS_H5H","objectType":"ManagedElement","objectUID":"5401HWWX1MNE3221291436","origSeverity":2,"rNeName":"","rNeType":"","rNeUID":"","specificProblem":"566","specificProblemID":"29810"}
# B = A["addInfo"]
# C = B.split(';')
# ll = [D for D in C if 'gNodeBId' in D]
# E = ll[0].split(':')[1]
# i = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\西藏AIOps样例数据\告警数据/'
# start = time.time()
# yy = pd.read_csv('{}/告警.csv'.format(i), iterator=True, encoding='utf-8')
# df = yy.get_chunk(20000)
# df.to_csv('{}/告警new.csv'.format(i),index=False, encoding='utf-8')


# import re
# x = '%1234acsa000dcsc---==976#'
# num3 = re.sub(u"([^\u0030-\u0039])", "", x)


# # 统计告警的情况
# datess = '20220514-20220615'
#
# alarm_path11 = '/data/alarm_hour_wuxiandonghuan/'
# # alarm_path = './Data/全省/Alert_Data/'
#
# date_low = datess.split('-')[0]
# date_high = datess.split('-')[-1]
# date = pd.date_range(date_low,date_high)
# date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()
#
# date_list_sum = []
# for date_day in date_list:
#     print(date_day)
#     try:date_data = pd.read_csv(alarm_path11+'/ALARM-{}.csv'.format(date_day),encoding='gbk')
#     except:date_data = pd.read_csv(alarm_path11+'/ALARM-{}.csv'.format(date_day),encoding='gb18030')
#
#     date_sele = date_data[['告警逻辑子类','告警标题','二级专业','告警标准名','设备厂家']]
#     date_sele_1 = date_sele[(date_sele['二级专业'] == '4G') |(date_sele['二级专业'] == '5G')]
#     date_list_sum.append(date_sele_1)
#
# date_PD = pd.concat(date_list_sum,axis=0)
# date_PD.to_csv('/home/AIOps/NM_Project/Data/汇总/统计.csv',index=False)

# # # ------------- 20220220 辽宁工单数据处理 --------------------------
# alert_data = pd.read_csv('C:/Users/x1carbon\Desktop\LN_服务器数据/order_20210915.csv',sep='|')
# alert_data.to_csv('C:/Users/x1carbon\Desktop\LN_服务器数据/order_20210915_new.csv',index=False)


# # ------------- 20220220 辽宁告警数据处理 --------------------------
# alert_data = pd.read_csv('C:/Users/x1carbon\Desktop\LN_服务器数据/wuxian_alarm_20220217.csv')
# alert_data.rename(columns={'告警级别': '网管告警级别', '网元名称': '基站名称', '县市': '地市名称', '专业': '网络类型',
#                            '基站编号': 'ENODEB_ID', '清除时间': '告警清除时间', '设备机房': '所属机房'}, inplace=True)
# alert_4G = alert_data[(alert_data['网络类型'] == '4G') | (alert_data['网络类型'] == '5G')]
# alert_4G['网络类型'] = alert_4G['网络类型'].map(lambda x: 103 if x == '4G' else 108)
#
# alert_dropna = alert_4G.dropna(subset=['基站名称', 'ENODEB_ID'])
# alert_dropna = alert_dropna[
#     ['网络类型', '设备类型', '基站名称', 'ENODEB_ID', '所属机房', '设备厂家名称','地市名称',
#      '网管告警级别', '告警标题', '告警发生时间', '告警清除状态', '告警清除时间', '设备工程状态', '告警工程状态']]
# alert_dropna.rename(columns={'基站名称':'网元名称'}, inplace=True)
#
# Data_org = alert_dropna.copy()
# try:
#     Data_org['告警持续时间1'] = Data_org['告警发生时间'].map(lambda x: int(time.mktime(time.strptime(x, "%Y/%m/%d %H:%M:%S"))))
#     Data_org['告警清除时间'] = Data_org['告警清除时间'].fillna(-1)
#     Data_org['告警持续时间2'] = Data_org['告警清除时间'].map(
#         lambda x: int(time.mktime(time.strptime(x, "%Y/%m/%d %H:%M:%S"))) if x != -1 else -1)
#     Data_org['告警持续时间'] = Data_org['告警持续时间2'] - Data_org['告警持续时间1']
#     Data_org['告警持续时间'] = Data_org['告警持续时间'].map(lambda x: x if x >= 0 else 9999)
#     Data_org = Data_org.drop(['告警持续时间1', '告警持续时间2'], axis=1)
# except:
#     Data_org['告警持续时间'] = 9999
# Data_org['告警日期'] = Data_org['告警发生时间'].map(lambda x: x.split()[0])
# Data_org['告警日期1'] = Data_org['告警日期'].map(
#     lambda x: datetime(int(x.split('/')[0]), int(x.split('/')[1]), int(x.split('/')[2])))
#
# # Data_target = Data_org[(Data_org['告警日期1'] >= file_date) & (Data_org['告警日期1'] <= file_date)]
# Alert_XJ = Data_org[['告警标题', '告警发生时间', '网元名称', 'ENODEB_ID', '设备厂家名称', '地市名称', '告警持续时间','网络类型']]
# Alert_XJ.columns = ['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间','网络类型']
# Alert_XJ.dropna(axis=0,subset = ['区域（地市）'],inplace=True)
# Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('地区')[0])
#
#
# # 告警标题里 '\/' 转为 '/'
# Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x:x.replace('\/','/'))
# try:Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x:x.replace('\\/','/'))
# except:print(11)

# ------------- 告警数据统计 --------------------------
# test_data = pd.read_csv('F:\AIOps_LN\智能基站运维数据-LN/LN_4G_HW.csv')
# gaojingbiaoti = test_data[['告警标题','设备厂家名称']]
# aaaa = test_data['告警标题'].value_counts()
# aaaa.to_csv('F:\AIOps_LN;\智能基站运维数据-LN/华为4G告警标题统计.csv')
# # gaojingbiaoti1 = gaojingbiaoti.drop_duplicates(subset = ['告警标题'])
# # gaojingbiaoti1.to_csv('F:\AIOps_ShanX\智能基站运维数据-LN/华为4G告警标题.csv')

# # # ------------- 按天分割告警文件 --------------------------
# # distr_list = ['西安', '咸阳', '延安', '榆林', '汉中', '宝鸡', '铜川', '安康', '商洛', '渭南']
# # for distr in distr_list:
# Data_org = pd.read_csv('F:\AIOps_LN\智能基站运维数据-LN/故障_LN_5G_HW_delJZ.csv', encoding='gbk')
# Data_org['告警日期'] = Data_org['告警开始时间'].map(lambda x:str(x).split(' ')[0].replace('/',''))
# date_low = '20210217'
# date_high = '20210816'
# date1 = pd.date_range(date_low, date_high)
# date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()
# for date_time in date_list:
#     date_select = Data_org[Data_org['告警日期']==date_time]
#     date_select.drop(['告警日期'],inplace=True,axis=1)
#     # date_select['网络类型'] = 103
#     # date_select.to_csv('E:\PycharmProjects\LZ_Project\Data\全省\Alert_Data/告警日志{}.csv'.format(date_time),index=False,encoding='gbk')
#     date_select.to_csv('F:\AIOps_LN\智能基站运维数据-LN/故障_LN_5G_HW_delJZ/HW_{}_delJZ_5G.csv'.format(date_time),index=False,encoding='gbk')
