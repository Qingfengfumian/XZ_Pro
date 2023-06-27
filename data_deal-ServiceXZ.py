# -*- coding: utf-8 -*-
import pandas as pd
import os
import time,os
import pandas as pd
import numpy as np
from datetime import datetime

from utils import mkdir
import warnings
from collections import Counter
from utils import real_warns_4G,real_warns_5G
#datess = '20211008-20220419'
datess = '20220208-20220430'

distr_list = ['全省']
ftype_list = ['华为','中兴']
alarm_path11 = '/data/Alarm_deal_data/'
# alarm_path11 = './Data/全省/Alert_Data/'
# alarm_path = './Data/全省/Alert_Data/'
from utils import params_setup
para = params_setup()
date_low = datess.split('-')[0]
date_high = datess.split('-')[-1]
date = pd.date_range(date_low,date_high)
date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()

def NeID_ana(x):
    try:alarm_dict = eval(x)
    except:return('空')
    alarm_info = alarm_dict["addInfo"]
    info_sp = alarm_info.split(';')
    NeID_tex = [D_E for D_E in info_sp if 'eNodeBId' in D_E]
    if len(NeID_tex) == 0:
        return('空')
    else:
        NeID_data = NeID_tex[0].split(':')[1]
        return NeID_data

def split_distr(alert_data,file_date,distr_list):
    # ---------- 西藏新增网元ID解析 数据处理 20220802--------------
    alarm_sele = alert_data[(alert_data['SPECIALTYEX2'] == '4G') | (alert_data['SPECIALTYEX2'] == '5G')]
    alert_data = alarm_sele[['NENAME', 'SPECIALTYEX2', 'EVENT_TIME', 'VENDOR', 'CANCEL_TIME', 'ALARMTITLE']]
    nm_date = pd.read_csv('./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv', encoding='gbk')
    nm_ID = nm_date[['基站名称', 'ENODEB_ID']]
    merge_data1 = pd.merge(alert_data, nm_ID, how='left', left_on='NENAME', right_on='基站名称')
    merge_data1.drop(columns=['基站名称'], inplace=True)
    nm_AREA = nm_date[['基站名称', '所属地市']]
    merge_data2 = pd.merge(merge_data1, nm_AREA, how='left', left_on='NENAME', right_on='基站名称')
    merge_data2.drop(columns=['基站名称'], inplace=True)
    alarm_sele = merge_data2.dropna()
    alert_data = alarm_sele.copy()

    alert_data.rename(columns = {'NENAME':'基站名称','所属地市':'地市名称','SPECIALTYEX2':'网络类型','EVENT_TIME':'告警发生时间','VENDOR':'设备厂家名称',
                                 'CANCEL_TIME':'告警清除时间','ALARMTITLE':'告警标题'},inplace=True)
    alert_4G = alert_data[(alert_data['网络类型'] == '4G') | (alert_data['网络类型'] == '5G')]
    alert_4G['网络类型'] = alert_4G['网络类型'].map(lambda x: 103 if x=='4G' else 108)
    # alert_4G['地市名称'] = '全省'
    # opt_select = opt_data[['基站名称', 'ENODEB_ID']]
    # opt_select.drop_duplicates(subset=['ENODEB_ID'], keep='first', inplace=True)
    #
    # alert_merge = pd.merge(alert_4G, opt_select, left_on='网元名称', right_on='ENODEB_ID', how='left')
    # alert_del = alert_merge.drop(columns=['ENODEB_ID'])

    alert_dropna = alert_4G.dropna(subset=['告警标题','地市名称','基站名称', 'ENODEB_ID'])
    # # 处理网元ID中的非数字字符
    # import re
    # alert_dropna['ENODEB_ID'] = alert_dropna['ENODEB_ID'].map(lambda x: re.sub(u"([^\u0030-\u0039])", "", str(x)))
    alert_dropna = alert_dropna[
        ['网络类型', '基站名称', 'ENODEB_ID', '设备厂家名称','地市名称',
         '告警标题', '告警发生时间', '告警清除时间']]

    alert_dropna.rename(columns={'基站名称':'网元名称'}, inplace=True)
    # 告警发生时间和清除时间处理 20220621
    try:alert_dropna['告警发生时间'] = alert_dropna['告警发生时间'].map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    except:alert_dropna['告警发生时间'] = alert_dropna['告警发生时间'].map(lambda x: datetime.strptime(x, "%Y/%m/%d %H:%M"))

    Data_org = alert_dropna.copy()
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
    Data_org['告警持续时间'] = 9999
    Data_org['告警日期'] = Data_org['告警发生时间'].map(lambda x: str(x).split()[0])
    Data_org['告警日期1'] = Data_org['告警日期'].map(
        lambda x: datetime(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))

    Data_target = Data_org[(Data_org['告警日期1'] >= file_date) & (Data_org['告警日期1'] <= file_date)]
    Alert_XJ = Data_target[['告警标题', '告警发生时间', '网元名称', 'ENODEB_ID', '设备厂家名称', '地市名称', '告警持续时间','网络类型']]
    Alert_XJ.columns = ['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间','网络类型']
    # Alert_XJ.dropna(axis=0,subset = ['区域（地市）'],inplace=True)
    Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('市')[0])
    Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('地区')[0])


    # 告警标题里 '\/' 转为 '/'
    Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x:x.replace('\/','/'))
    try:Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x:x.replace('\\/','/'))
    except:print(11)
    for i_region in distr_list:
        # Alert_region = Alert_XJ[Alert_XJ['区域（地市）'] == i_region]
        Alert_region = Alert_XJ
        region_path = "./Data/{}/Alert_Data/".format(i_region)
        if len(Alert_region) > 0:
            if os.path.exists(region_path) == False:
                mkdir(region_path)
            Alert_region.to_csv(
                region_path + '告警日志{}.csv'.format(file_date), index=False,encoding='gbk')
        else:
            print('Warning: Alert data in {}-{} is None\n Please check!!'.format(i_region,file_date))
def data_del_part_4G(para,alert_data_part,tr_date, i_region):
    """
    :param grp_cnt: 每分钟告警数量异常阈值 例如：1500
    :param del_alm: 每分钟告警数量异常，删除告警类型 例如：['用户面承载链路故障告警','时钟参考源异常告警']
    :param del_min3: 删除告警持续时间小于3分钟的告警  例如：同上
    :param sloc: 读入数据时，用于定位的字段 例如：'级别'
    :param fatr: 数据所属厂家 例如：'华为'
    :return: Data_delJZ：未删除夜间告警
            Data：删除夜间告警
    """
    grp_cnt = para.grp_cnt
    del_alm = para.del_alm
    del_min3 = para.del_min3
    Data_org = alert_data_part[alert_data_part['网络类型']==103]
    date = tr_date
    distr = i_region
    Data_org = Data_org[['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '告警持续时间','区域（地市）']]
    Data_Org1 = Data_org

    Data_org2 = Data_Org1
    Data_org3 = Data_org2[~Data_org2['基站id'].isin(['-',' -',' '])]  # 空白id = '-'

    Data_org3['告警开始时间'] = Data_org3['告警开始时间'].map(lambda x: str(x).replace('-', '/'))
    Data_org3 = Data_org3.sort_values(by='告警开始时间')
# -------------------------- 删除异常时间点的告警 ------------------------
    # 剔除非影响业务告警
    def del_unimp(Data_org):
        Data_org41 = Data_org
        HW_alarmname = Data_org41['告警名称'].values.tolist()
        HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/剔除告警标题.csv', encoding='GBK',engine='python')
        HW_Per_list = np.ravel(HW_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
        if i_region == '渭南':
            HW_Per_list = HW_Per_list + ['用户面承载链路故障告警','模块故障']
        HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
        HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['告警名称'])  # 剔除+去重
        Data_org4 = pd.merge(Data_org41, HW_fault_Pd)
        Data_org4 = Data_org4.sort_values(by='告警开始时间')
        XN_alarmname = Data_org4['告警名称'].values.tolist()
        return Data_org4,XN_alarmname
    # 剔除性能告警
    def del_xn(Data_org4,XN_alarmname):
        XN_Per = pd.read_csv(r'./Data/汇总/Project_Opt/性能告警标题.csv', encoding='GBK',engine='python')
        XN_Per_list = np.ravel(XN_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
        Del_XN_list = list(set(XN_alarmname).difference(set(XN_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
        Del_XN_Pd = pd.DataFrame(Del_XN_list, columns=['告警名称'])  # 剔除+去重
        Data_org5 = pd.merge(Data_org4, Del_XN_Pd)
        Data_org5 = Data_org5.sort_values(by='告警开始时间')
        Data_org5['告警开始分钟'] = Data_org5['告警开始时间'].map(lambda x: x.split(':')[0] + ':' + x.split(':')[1])
        return Data_org5
    # 剔除告警持续时间小于3分钟的告警
    def del_mins(Data_org5):
        Group_mini = Data_org5.groupby('告警开始分钟').count()
        Group_count = Group_mini[Group_mini['告警开始时间'] > grp_cnt]
        del_mini = Group_count.index.tolist()
        Data_org6 = Data_org5
        for i in del_mini:
            if del_alm:
                del_lis = '|'.join(del_alm)
                Data_org6=Data_org6[~(Data_org6['告警开始时间'].str.contains(i) & (Data_org6['告警名称'].str.contains(del_lis)))]
            else:
                Data_org6 = Data_org6[~(Data_org6['告警开始时间'].str.contains(i))]  # 汉中

        Data_org6 = Data_org6.drop(['告警开始分钟'], axis=1)
        Data_org7 = Data_org6[~((Data_org6['告警持续时间'] <= 180) & (Data_org6['告警名称'].isin(del_min3)))]
        # Data_org8 = Data_org7.copy()
        # Data_org8['制式'] = '4G'
        # Data_org8 = Data_org8[['告警开始时间', '告警结束时间', '告警名称', '基站id', '网元名称', '制式', '厂家']]
        return Data_org7
    # 剔除每日退服
    def del_daily(Data_org7):
        Data_org9 = Data_org7.copy()
        HW_alarmname = Data_org9['网元名称'].values.tolist()
        if os.path.exists(r'./Data/汇总/Project_Opt/每天退服基站.csv'.format(distr)):
            HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/每天退服基站.csv'.format(distr), encoding='GBK',engine='python')
            HW_Per_list = np.ravel(HW_Per.values).tolist()
            HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集
            HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['网元名称'])
            Data_org10 = pd.merge(Data_org9, HW_fault_Pd, on='网元名称')
            Data_org11 = Data_org10.sort_values(by='告警开始时间')
        else:
            print('./Data/汇总/Project_Opt/每天退服基站.csv Not Existed!'.format(distr))
            Data_org11 = Data_org9
        return Data_org11
    # 剔除9-11 17-19载波调度站 发生网元连接中断告警 且1分钟内恢复的
    def del_zb(Data_org11):
        Data_org11 = Data_org11.copy()
        if os.path.exists(r'./Data/{}/Project_Opt/载波调度清单.xlsx'.format(distr)):
            Car_sch = pd.read_excel(r'./Data/{}/Project_Opt/载波调度清单.xlsx'.format(distr))
            Data_org11['告警开始小时'] = Data_org11['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
            Data_org11['告警持续时间'] = Data_org11['告警持续时间'].map(lambda x: int(x))
            Data_org12 = Data_org11[~(Data_org11['网元名称'].isin(Car_sch['基站名称']) & Data_org11['告警开始小时'].isin(
                ['09', '10', '11', '17', '18', '19']) & (Data_org11['告警持续时间'] <= 60))]
            Data_org13 = Data_org12.copy()
            Data_org13['制式'] = '4G'
            Data_org13 = Data_org13[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家','区域（地市）']]
        else:
            print(r'./Data/{}/Project_Opt/载波调度清单.xlsx Not Existed!'.format(distr))
            Data_org13 = Data_org11
        return Data_org13
    # 剔除夜间告警
    def del_night(Data_org13):
        Data_org14 = Data_org13
        Data_org14['告警开始小时'] = Data_org14['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
        Data_org14 = Data_org14[~Data_org14['告警开始小时'].isin(['23', '00', '01', '02', '03', '04', '05', '06'])]
        Data_org14 = Data_org14.drop(['告警开始小时'], axis=1)
        return Data_org14

    def max_tfwarn(Data_org):
        Data_org['告警开始时间1'] = pd.to_datetime(Data_org['告警开始时间'])
        Data_org['date'] = pd.to_datetime(Data_org['告警开始时间1'].dt.date)
        # Data_org = Data_org.groupby('date')
        sum_id = len(set(Data_org['基站id']))
        Data_org["data_hour"] = pd.to_datetime(Data_org['告警开始时间']).dt.hour
        date = Data_org.iloc[0]['date']
        TF_org = Data_org[Data_org['告警名称'].isin(real_warns_4G)]
        TF_org1 = TF_org.drop_duplicates(subset=['基站id', 'data_hour'])
        tmp = Counter(TF_org1['data_hour'])
        tmp = sorted(tmp.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        try:
            max_hour = tmp[0][0]
        except:
            return Data_org
        tf_id = len(set(TF_org['基站id']))
        if tf_id >= (0.8 * sum_id):
            Data_org1 = Data_org[~((Data_org['data_hour'] == max_hour) & (Data_org['告警名称'].isin(real_warns_4G)))]
            print('{}退服基站数量异常{}/{},{}：00退服基站{},删除告警{}条'.format(date, tf_id, sum_id, max_hour, tmp[0][1],
                                                               Data_org.shape[0] - Data_org1.shape[0]))
        else:
            Data_org1 = Data_org
        Data_org1.drop(columns=['告警开始时间1', 'date', 'data_hour'])
        return Data_org1


    Data_org4, XN_alarmname = del_unimp(Data_org3)
    Data_org5 = del_xn(Data_org4,XN_alarmname)
    Data_org7 = del_mins(Data_org5)
    Data_org11 = del_daily(Data_org7)
    Data_org13 = del_zb(Data_org11)
    Data_org14 = del_night(Data_org13)

    tf_alarm = Data_org3['告警名称'].values.tolist()
    Data_tf = del_xn(Data_org3,tf_alarm)
    Data_tf = max_tfwarn(Data_tf)
    Data_tf = del_mins(Data_tf)
    # if distr in ['铜川','汉中']:
    #     Data_tf = del_daily(Data_tf)
    Data_tf = del_zb(Data_tf)
    region_del_path = './Data/{}/Alert_Data/'.format(distr)
    if os.path.exists(region_del_path) == False:
        mkdir(region_del_path)

    Data_tf.to_csv(r'./Data/{}/Alert_Data/故障_{}_delJZ_4G.csv'.format(distr, date), index=None, encoding='gbk')
    Data_delJZ = Data_tf

    Data_org14.to_csv(r'./Data/{}/Alert_Data/故障_{}_4G.csv'.format(distr,date), index=None,encoding='gbk')
    Data = Data_org14
    return Data_delJZ,Data, date, distr
def data_del_part_5G(para,alert_data_part,tr_date, i_region):
    """
    :param grp_cnt: 每分钟告警数量异常阈值 例如：1500
    :param del_alm: 每分钟告警数量异常，删除告警类型 例如：['用户面承载链路故障告警','时钟参考源异常告警']
    :param del_min3: 删除告警持续时间小于3分钟的告警  例如：同上
    :param sloc: 读入数据时，用于定位的字段 例如：'级别'
    :param fatr: 数据所属厂家 例如：'华为'
    :return: Data_delJZ：未删除夜间告警
            Data：删除夜间告警
    """
    grp_cnt = para.grp_cnt
    del_alm = para.del_alm
    del_min3 = para.del_min3
    Data_org = alert_data_part[alert_data_part['网络类型']==108]
    if Data_org.empty:
        # Data_org_20 = pd.read_csv(r'./Data/{}/Alert_Data/故障_{}_4G.csv'.format(i_region, tr_date), encoding='gbk')
        # Data_org_20.to_csv(r'./Data/{}/Alert_Data/故障_{}_5G.csv'.format(i_region, tr_date), index=None, encoding='gbk')
        # Data_org_30 = pd.read_csv(r'./Data/{}/Alert_Data/故障_{}_delJZ_4G.csv'.format(i_region, tr_date), encoding='gbk')
        # Data_org_30.to_csv(r'./Data/{}/Alert_Data/故障_{}_delJZ_5G.csv'.format(i_region, tr_date), index=None, encoding='gbk')
        print('{}-{}5G数据为空'.format(i_region,tr_date))
        return (0)

    date = tr_date
    distr = i_region
    Data_org = Data_org[['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '告警持续时间','区域（地市）']]
    Data_Org1 = Data_org

    Data_org2 = Data_Org1
    Data_org3 = Data_org2[~Data_org2['基站id'].isin(['-',' -',' '])]  # 空白id = '-'

    Data_org3['告警开始时间'] = Data_org3['告警开始时间'].map(lambda x: str(x).replace('-', '/'))
    Data_org3 = Data_org3.sort_values(by='告警开始时间')
# -------------------------- 删除异常时间点的告警 ------------------------
    # 剔除非影响业务告警
    def del_unimp(Data_org):
        Data_org41 = Data_org
        HW_alarmname = Data_org41['告警名称'].values.tolist()
        HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/剔除告警标题.csv', encoding='GBK',engine='python')
        HW_Per_list = np.ravel(HW_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
        if i_region == '渭南':
            HW_Per_list = HW_Per_list + ['用户面承载链路故障告警','模块故障']
        HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
        HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['告警名称'])  # 剔除+去重
        Data_org4 = pd.merge(Data_org41, HW_fault_Pd)
        Data_org4 = Data_org4.sort_values(by='告警开始时间')
        XN_alarmname = Data_org4['告警名称'].values.tolist()
        return Data_org4,XN_alarmname
    # 剔除性能告警
    def del_xn(Data_org4,XN_alarmname):
        XN_Per = pd.read_csv(r'./Data/汇总/Project_Opt/性能告警标题.csv', encoding='GBK',engine='python')
        XN_Per_list = np.ravel(XN_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
        Del_XN_list = list(set(XN_alarmname).difference(set(XN_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
        Del_XN_Pd = pd.DataFrame(Del_XN_list, columns=['告警名称'])  # 剔除+去重
        Data_org5 = pd.merge(Data_org4, Del_XN_Pd)
        Data_org5 = Data_org5.sort_values(by='告警开始时间')
        Data_org5['告警开始分钟'] = Data_org5['告警开始时间'].map(lambda x: x.split(':')[0] + ':' + x.split(':')[1])
        return Data_org5
    # 剔除告警持续时间小于3分钟的告警
    def del_mins(Data_org5):
        Group_mini = Data_org5.groupby('告警开始分钟').count()
        Group_count = Group_mini[Group_mini['告警开始时间'] > grp_cnt]
        del_mini = Group_count.index.tolist()
        Data_org6 = Data_org5
        for i in del_mini:
            if del_alm:
                del_lis = '|'.join(del_alm)
                Data_org6=Data_org6[~(Data_org6['告警开始时间'].str.contains(i) & (Data_org6['告警名称'].str.contains(del_lis)))]
            else:
                Data_org6 = Data_org6[~(Data_org6['告警开始时间'].str.contains(i))]  # 汉中

        Data_org6 = Data_org6.drop(['告警开始分钟'], axis=1)
        Data_org7 = Data_org6[~((Data_org6['告警持续时间'] <= 180) & (Data_org6['告警名称'].isin(del_min3)))]
        # Data_org8 = Data_org7.copy()
        # Data_org8['制式'] = '4G'
        # Data_org8 = Data_org8[['告警开始时间', '告警结束时间', '告警名称', '基站id', '网元名称', '制式', '厂家']]
        return Data_org7
    # 剔除每日退服
    def del_daily(Data_org7):
        Data_org9 = Data_org7.copy()
        HW_alarmname = Data_org9['网元名称'].values.tolist()
        if os.path.exists(r'./Data/汇总/Project_Opt/每天退服基站.csv'.format(distr)):
            HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/每天退服基站.csv'.format(distr), encoding='GBK',engine='python')
            HW_Per_list = np.ravel(HW_Per.values).tolist()
            HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集
            HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['网元名称'])
            Data_org10 = pd.merge(Data_org9, HW_fault_Pd, on='网元名称')
            Data_org11 = Data_org10.sort_values(by='告警开始时间')
        else:
            print('./Data/汇总/Project_Opt/每天退服基站.csv Not Existed!'.format(distr))
            Data_org11 = Data_org9
        return Data_org11
    # 剔除9-11 17-19载波调度站 发生网元连接中断告警 且1分钟内恢复的
    def del_zb(Data_org11):
        Data_org11 = Data_org11.copy()
        if os.path.exists(r'./Data/{}/Project_Opt/载波调度清单.xlsx'.format(distr)):
            Car_sch = pd.read_excel(r'./Data/{}/Project_Opt/载波调度清单.xlsx'.format(distr))
            Data_org11['告警开始小时'] = Data_org11['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
            Data_org11['告警持续时间'] = Data_org11['告警持续时间'].map(lambda x: int(x))
            Data_org12 = Data_org11[~(Data_org11['网元名称'].isin(Car_sch['基站名称']) & Data_org11['告警开始小时'].isin(
                ['09', '10', '11', '17', '18', '19']) & (Data_org11['告警持续时间'] <= 60))]
            Data_org13 = Data_org12.copy()
            Data_org13['制式'] = '5G'
            Data_org13 = Data_org13[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家','区域（地市）']]
        else:
            print(r'./Data/{}/Project_Opt/载波调度清单.xlsx Not Existed!'.format(distr))
            Data_org13 = Data_org11
        return Data_org13
    # 剔除夜间告警
    def del_night(Data_org13):
        Data_org14 = Data_org13
        Data_org14['告警开始小时'] = Data_org14['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
        Data_org14 = Data_org14[~Data_org14['告警开始小时'].isin(['23', '00', '01', '02', '03', '04', '05', '06'])]
        Data_org14 = Data_org14.drop(['告警开始小时'], axis=1)
        return Data_org14

    def max_tfwarn(Data_org):
        Data_org['告警开始时间1'] = pd.to_datetime(Data_org['告警开始时间'])
        Data_org['date'] = pd.to_datetime(Data_org['告警开始时间1'].dt.date)
        # Data_org = Data_org.groupby('date')
        sum_id = len(set(Data_org['基站id']))
        Data_org["data_hour"] = pd.to_datetime(Data_org['告警开始时间']).dt.hour
        date = Data_org.iloc[0]['date']
        TF_org = Data_org[Data_org['告警名称'].isin(real_warns_5G)]
        TF_org1 = TF_org.drop_duplicates(subset=['基站id', 'data_hour'])
        tmp = Counter(TF_org1['data_hour'])
        tmp = sorted(tmp.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        try:
            max_hour = tmp[0][0]
        except:
            return Data_org
        tf_id = len(set(TF_org['基站id']))
        if tf_id >= (0.8 * sum_id):
            Data_org1 = Data_org[~((Data_org['data_hour'] == max_hour) & (Data_org['告警名称'].isin(real_warns_5G)))]
            print('{}退服基站数量异常{}/{},{}：00退服基站{},删除告警{}条'.format(date, tf_id, sum_id, max_hour, tmp[0][1],
                                                               Data_org.shape[0] - Data_org1.shape[0]))
        else:
            Data_org1 = Data_org
        Data_org1.drop(columns=['告警开始时间1', 'date', 'data_hour'])
        return Data_org1


    Data_org4, XN_alarmname = del_unimp(Data_org3)
    Data_org5 = del_xn(Data_org4,XN_alarmname)
    Data_org7 = del_mins(Data_org5)
    Data_org11 = del_daily(Data_org7)
    Data_org13 = del_zb(Data_org11)
    Data_org14 = del_night(Data_org13)

    tf_alarm = Data_org3['告警名称'].values.tolist()
    Data_tf = del_xn(Data_org3,tf_alarm)
    Data_tf = max_tfwarn(Data_tf)
    Data_tf = del_mins(Data_tf)
    # if distr in ['铜川','汉中']:
    #     Data_tf = del_daily(Data_tf)
    Data_tf = del_zb(Data_tf)
    region_del_path = './Data/{}/Alert_Data/'.format(distr)
    if os.path.exists(region_del_path) == False:
        mkdir(region_del_path)

    Data_tf.to_csv(r'./Data/{}/Alert_Data/故障_{}_delJZ_5G.csv'.format(distr, date), index=None, encoding='gbk')
    Data_delJZ = Data_tf

    Data_org14.to_csv(r'./Data/{}/Alert_Data/故障_{}_5G.csv'.format(distr,date), index=None,encoding='gbk')
    Data = Data_org14
    return Data_delJZ,Data, date, distr
def split_ftype_4G(i_region,tr_date,ftype_list):
    try:
        alert_data1 = pd.read_csv("./Data/{}/Alert_Data/故障_{}_4G.csv".format(i_region, tr_date),encoding='gbk')
        alert_data_del1 = pd.read_csv("./Data/{}/Alert_Data/故障_{}_delJZ_4G.csv".format(i_region, tr_date),encoding='gbk')
    except:
        print("Error: 没有找到文件或读取文件失败:{}".format(
            "./Data/{}/Alert_Data/故障_{}_4G.csv".format(i_region, tr_date)))
        return (0)  # 20220606 跳出函数 继续执行下一天的操作
        os._exit(0)
    for i_ftype in ftype_list:
        alert_data = alert_data1[alert_data1['厂家'] == i_ftype]
        alert_data_del = alert_data_del1[alert_data_del1['厂家'] == i_ftype]
        region_path ='./Data/{}/Alert_Deal/Samp_{}/'.format(i_region, i_ftype)
        if len(alert_data) > 0:
            if os.path.exists(region_path) == False:
                mkdir(region_path)
            alert_data.to_csv(region_path + '故障_处理_{}_4G.csv'.format(tr_date), index=False,encoding='gbk')
            alert_data_del.to_csv(region_path + '故障_处理_{}_delJZ_4G.csv'.format(tr_date), index=False,encoding='gbk')
        else:
            print('Warning: Alert data in {}-{}-{}_4G is None!!'.format(i_region, i_ftype,tr_date))
def split_ftype_5G(i_region,tr_date,ftype_list):
    try:
        alert_data1 = pd.read_csv("./Data/{}/Alert_Data/故障_{}_5G.csv".format(i_region, tr_date),encoding='gbk')
        alert_data_del1 = pd.read_csv("./Data/{}/Alert_Data/故障_{}_delJZ_5G.csv".format(i_region, tr_date),encoding='gbk')
    except:
        print("Error: 没有找到文件或读取文件失败:{}".format(
            "./Data/{}/Alert_Data/故障_{}_5G.csv".format(i_region, tr_date)))
        return(0) # 20220606 跳出函数 继续执行下一天的操作
        os._exit(0)
    for i_ftype in ftype_list:
        alert_data = alert_data1[alert_data1['厂家'] == i_ftype]
        alert_data_del = alert_data_del1[alert_data_del1['厂家'] == i_ftype]
        region_path ='./Data/{}/Alert_Deal/Samp_{}/'.format(i_region, i_ftype)
        if len(alert_data) > 0:
            if os.path.exists(region_path) == False:
                mkdir(region_path)
            alert_data.to_csv(region_path + '故障_处理_{}_5G.csv'.format(tr_date), index=False,encoding='gbk')
            alert_data_del.to_csv(region_path + '故障_处理_{}_delJZ_5G.csv'.format(tr_date), index=False,encoding='gbk')
        else:
            print('Warning: Alert data in {}-{}-{}_5G is None!!'.format(i_region, i_ftype,tr_date))

for tr_date in date_list:
    print(tr_date)
    run_split_first = True
    for i_region in distr_list:
        alarm_path = "./Data/{}/Alert_Data/告警日志{}.csv".format(i_region, tr_date)
        # print(alarm_path)
        if (os.path.exists(alarm_path)==False) & (run_split_first==True):
            try:
                alert_data = pd.read_csv("{}/ALARM-{}.csv".format(alarm_path11,tr_date),encoding='gbk')
            except:
                try:
                    alert_data = pd.read_csv("{}/ALARM-{}.csv".format(alarm_path11, tr_date),encoding='gb18030')
                except:
                    print("Error: 没有找到文件或读取文件失败:{}".format("{}/ALARM-{}.csv".format(alarm_path11, tr_date)))
                    os._exit(0)
            split_distr(alert_data,tr_date,distr_list)
            run_split_first = False
        else:
            continue
    for i_region in distr_list:
        deal_path = "./Data/{}/Alert_Data/故障_{}_4G.csv".format(i_region, tr_date)
        if os.path.exists(deal_path) == False:
            try:
                alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region,tr_date),encoding='gbk',engine='python')
            except:
                print("Error: 没有找到文件或读取文件失败:{}".format("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region,tr_date)))
                os._exit(0)
            data_del_part_4G(para,alert_data_part,tr_date,i_region)
        deal_path_5G = "./Data/{}/Alert_Data/故障_{}_5G.csv".format(i_region, tr_date)
        if os.path.exists(deal_path_5G) == False:
            try:
                alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region, tr_date),encoding='gbk', engine='python')
            except:
                print("Error: 没有找到文件或读取文件失败:{}".format("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region, tr_date)))
                os._exit(0)
            data_del_part_5G(para, alert_data_part, tr_date, i_region)
    for i_region in distr_list:
        ftype_split_first = True
        for ftype in ftype_list:
            ftype_path = './Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_4G.csv'.format(i_region,ftype, tr_date)
            if (os.path.exists(ftype_path) == False) & (ftype_split_first==True):
                split_ftype_4G(i_region,tr_date,ftype_list)
                ftype_split_first = False
        ftype_split_first_5G = True
        for ftype in ftype_list:
            ftype_path_5G = './Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_5G.csv'.format(i_region,ftype, tr_date)
            if (os.path.exists(ftype_path_5G) == False) & (ftype_split_first_5G==True):
                split_ftype_5G(i_region,tr_date,ftype_list)
                ftype_split_first_5G = False
