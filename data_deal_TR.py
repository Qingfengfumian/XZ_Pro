import time,os
import pandas as pd
import numpy as np
from datetime import datetime

from utils import mkdir
import warnings
from collections import Counter
from utils import real_warns_4G,real_warns_5G
warnings.filterwarnings('ignore')


def split_distr(alert_data,file_date,distr_list):
    # alert_data.columns = ['一级网络类型', '网络类型', '设备类型', '网元名称', '所属机房', '设备厂家名称', '设备名称', '地市名称', '告警对象名称',
    #                       '告警对象设备类型', '网管告警级别', '告警标题', '告警发生时间', '告警清除状态', '告警清除时间', '设备工程状态', '告警工程状态']
    alert_4G = alert_data[(alert_data['网络类型'] == 103) | (alert_data['网络类型'] == 108)]
    # alert_4G['ENODEB_ID'] = alert_4G['基站名称']
    # opt_select = opt_data[['基站名称', 'ENODEB_ID']]
    # opt_select.drop_duplicates(subset=['ENODEB_ID'], keep='first', inplace=True)
    #
    # alert_merge = pd.merge(alert_4G, opt_select, left_on='网元名称', right_on='ENODEB_ID', how='left')
    # alert_del = alert_merge.drop(columns=['ENODEB_ID'])

    alert_dropna = alert_4G.dropna(subset=['基站名称', 'ENODEB_ID'])

    alert_dropna = alert_dropna[
        ['一级网络类型', '网络类型', '设备类型', '基站名称', 'ENODEB_ID', '所属机房', '设备厂家名称', '设备名称', '地市名称', '告警对象名称', '告警对象设备类型',
         '网管告警级别', '告警标题', '告警发生时间', '告警清除状态', '告警清除时间', '设备工程状态', '告警工程状态']]
    alert_dropna.rename(columns={'基站名称':'网元名称'}, inplace=True)

    Data_org = alert_dropna.copy()
    try:
        Data_org['告警持续时间1'] = Data_org['告警发生时间'].map(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
        Data_org['告警清除时间'] = Data_org['告警清除时间'].fillna(-1)
        Data_org['告警持续时间2'] = Data_org['告警清除时间'].map(
            lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))) if x != -1 else -1)
        Data_org['告警持续时间'] = Data_org['告警持续时间2'] - Data_org['告警持续时间1']
        Data_org['告警持续时间'] = Data_org['告警持续时间'].map(lambda x: x if x >= 0 else 9999)
        Data_org = Data_org.drop(['告警持续时间1', '告警持续时间2'], axis=1)
    except:
        Data_org['告警持续时间'] = 9999
    Data_org['告警日期'] = Data_org['告警发生时间'].map(lambda x: x.split()[0])
    Data_org['告警日期1'] = Data_org['告警日期'].map(
        lambda x: datetime(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))

    Data_target = Data_org[(Data_org['告警日期1'] >= file_date) & (Data_org['告警日期1'] <= file_date)]
    Alert_XJ = Data_target[['告警标题', '告警发生时间', '网元名称', 'ENODEB_ID', '设备厂家名称', '地市名称', '告警持续时间','网络类型']]
    Alert_XJ.columns = ['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间','网络类型']
    Alert_XJ.dropna(axis=0,subset = ['区域（地市）'],inplace=True)
    Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('地区')[0])

    # 告警标题里 '\/' 转为 '/'
    Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x:x.replace('\/','/'))
    try:Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x:x.replace('\\/','/'))
    except:print(11)
    for i_region in distr_list:
        Alert_region = Alert_XJ[Alert_XJ['区域（地市）'] == i_region]
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
    Data_org = Data_org[['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '告警持续时间']]
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
        HW_Per_list = np.ravel(HW_Per['alarmname'].values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
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
            HW_Per_list = np.ravel(HW_Per['alarmname'].values).tolist()
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
            Data_org13 = Data_org13[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家']]
        else:
            print(r'./Data/{}/Project_Opt/载波调度清单.xlsx Not Existed!'.format(distr))
            Data_org13 = Data_org11
            Data_org13['制式'] = '4G'
            Data_org13 = Data_org13[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家']]
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
        if tf_id >= (0.6 * sum_id):
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
        Data_org_20 = pd.read_csv(r'./Data/{}/Alert_Data/故障_{}_4G.csv'.format(i_region, tr_date), encoding='gbk')
        Data_org_20.to_csv(r'./Data/{}/Alert_Data/故障_{}_5G.csv'.format(i_region, tr_date), index=None, encoding='gbk')
        Data_org_30 = pd.read_csv(r'./Data/{}/Alert_Data/故障_{}_delJZ_4G.csv'.format(i_region, tr_date), encoding='gbk')
        Data_org_30.to_csv(r'./Data/{}/Alert_Data/故障_{}_delJZ_5G.csv'.format(i_region, tr_date), index=None, encoding='gbk')
        print('{}-{}5G数据为空'.format(i_region,tr_date))
        return (0)

    date = tr_date
    distr = i_region
    Data_org = Data_org[['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '告警持续时间']]
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
        HW_Per_list = np.ravel(HW_Per['alarmname'].values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
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
            HW_Per_list = np.ravel(HW_Per['alarmname'].values).tolist()
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
            Data_org13 = Data_org13[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家']]
        else:
            print(r'./Data/{}/Project_Opt/载波调度清单.xlsx Not Existed!'.format(distr))
            Data_org13 = Data_org11
            Data_org13['制式'] = '5G'
            Data_org13 = Data_org13[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家']]
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
        if tf_id >= (0.6 * sum_id):
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
            print('Warning: Alert data in {}-{}-{} is None!!'.format(i_region, i_ftype,tr_date))
def split_ftype_5G(i_region,tr_date,ftype_list):
    try:
        alert_data1 = pd.read_csv("./Data/{}/Alert_Data/故障_{}_5G.csv".format(i_region, tr_date),encoding='gbk')
        alert_data_del1 = pd.read_csv("./Data/{}/Alert_Data/故障_{}_delJZ_5G.csv".format(i_region, tr_date),encoding='gbk')
    except:
        print("Error: 没有找到文件或读取文件失败:{}".format("./Data/{}/Alert_Data/故障_{}_5G.csv".format(i_region, tr_date)))
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
            print('Warning: Alert data in {}-{}-{} is None!!'.format(i_region, i_ftype,tr_date))
            alert_data = alert_data1
            alert_data_del = alert_data_del1
            alert_data.to_csv(region_path + '故障_处理_{}_5G.csv'.format(tr_date), index=False, encoding='gbk')
            alert_data_del.to_csv(region_path + '故障_处理_{}_delJZ_5G.csv'.format(tr_date), index=False, encoding='gbk')

def distr_del_cat_TR(para):
    distr_list = para.distr_list
    ftype_list = para.ftype_list
    if 'train' in para.mode:
        if para.mode == "train_XJ":
            dates = para.train_date
        if para.mode == "train_TF":
            dates = para.train_date1
    if para.mode == "predict":
        dates = para.date
    date_low = dates.split('-')[0]
    date_high = dates.split('-')[-1]
    date = pd.date_range(date_low,date_high)
    date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()
    for tr_date in date_list:
        print(tr_date)
        # run_split_first = True
        # for i_region in distr_list:
        #     alarm_path = "./Data/{}/Alert_Data/告警日志{}.csv".format(i_region, tr_date)
        #     if (os.path.exists(alarm_path)==False) & (run_split_first==True):
        #         try:
        #             alert_data = pd.read_csv("{}/alarm_{}_time.csv".format(para.alarm_path,tr_date))
        #         except:
        #             print("Error: 没有找到文件或读取文件失败:{}".format("{}/alarm_{}_time.csv".format(para.alarm_path,tr_date)))
        #             os._exit(0)
        #         split_distr(alert_data,tr_date,distr_list)
        #         run_split_first = False
        #     else:
        #         continue
        # for i_region in distr_list:
        #     deal_path = "./Data/{}/Alert_Data/故障_{}_4G.csv".format(i_region, tr_date)
        #     if os.path.exists(deal_path) == False:
        #         try:
        #             alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region,tr_date),encoding='gbk',engine='python')
        #         except:
        #             print("Error: 没有找到文件或读取文件失败:{}".format("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region,tr_date)))
        #             os._exit(0)
        #         data_del_part_4G(para,alert_data_part,tr_date,i_region)
        #     deal_path_5G = "./Data/{}/Alert_Data/故障_{}_5G.csv".format(i_region, tr_date)
        #     if os.path.exists(deal_path_5G) == False:
        #         try:
        #             alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region, tr_date),encoding='gbk', engine='python')
        #         except:
        #             print("Error: 没有找到文件或读取文件失败:{}".format("./Data/{}/Alert_Data/告警日志{}.csv".format(i_region, tr_date)))
        #             os._exit(0)
        #         data_del_part_5G(para, alert_data_part, tr_date, i_region)
        # for i_region in distr_list:
            # ftype_split_first = True
            # for ftype in ftype_list:
            #     ftype_path = './Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_4G.csv'.format(i_region,ftype, tr_date)
            #     if (os.path.exists(ftype_path) == False) & (ftype_split_first==True):
            #         split_ftype_4G(i_region,tr_date,ftype_list)
            #         ftype_split_first = False
            # ftype_split_first_5G = True
            # for ftype in ftype_list:
            #     ftype_path_5G = './Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_5G.csv'.format(i_region,ftype, tr_date)
            #     if (os.path.exists(ftype_path_5G) == False) & (ftype_split_first_5G==True):
            #         split_ftype_5G(i_region,tr_date,ftype_list)
            #         ftype_split_first_5G = False
    concat_date(para)
def concat_date(para):
    mode = para.mode
    distr_list = para.distr_list
    ftype = para.ftype
    if mode == "train_XJ":
        date = para.train_date
    elif mode == 'train_TF':
        date = para.train_date1
    elif mode == 'predict':
        date = para.date
    else:
        return ('ERROR:Wrong run mode!')
    start = int(date.split('-')[0])
    end = int(date.split('-')[-1])

    for distr in distr_list:
        tempall_4G = []
        temp2all_4G = []
        tempall_5G = []
        temp2all_5G = []
        for file in os.listdir(r'./Data/{}/Alert_Deal/Samp_{}/'.format(distr, ftype)):
            if 'delJZ' not in file:
                file_date = file.split('处理_')[-1].split('_')[0]
                if (int(file_date) >= start) & (int(file_date) <= end):
                    if '4G' in file:
                        temp = pd.read_csv(
                            r'./Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_4G.csv'.format(distr, ftype, file_date), encoding='gbk',
                            engine='python')
                        temp2 = pd.read_csv(
                            r'./Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_delJZ_4G.csv'.format(distr, ftype, file_date), encoding='gbk',
                            engine='python')
                        tempall_4G.append(temp)
                        temp2all_4G.append(temp2)
                    if '5G' in file:
                        temp3 = pd.read_csv(
                            r'./Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_5G.csv'.format(distr, ftype, file_date), encoding='gbk',
                            engine='python')
                        temp4 = pd.read_csv(
                            r'./Data/{}/Alert_Deal/Samp_{}/故障_处理_{}_delJZ_5G.csv'.format(distr, ftype, file_date), encoding='gbk',
                            engine='python')
                        tempall_5G.append(temp3)
                        temp2all_5G.append(temp4)
            else:
                continue
        AlertHW_DF2 = pd.concat(tempall_4G, axis=0)
        AlertHW_DF22 = pd.concat(temp2all_4G, axis=0)
        samp_path = './Data/{}/Alert_Deal/Samp_{}_{}/'.format(distr,mode, ftype)
        if os.path.exists(samp_path)==False:
            mkdir(samp_path)
        AlertHW_DF2['告警名称'] = AlertHW_DF2['告警名称'].map(lambda x: x.replace('\/', '/'))
        AlertHW_DF2.to_csv('./Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_4G.csv'.format(distr,mode, ftype, date),index=False,encoding='gbk')
        AlertHW_DF22.to_csv('./Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_delJZ_4G.csv'.format(distr,mode, ftype, date),index=False,encoding='gbk')

        AlertHW_DF3 = pd.concat(tempall_5G, axis=0)
        AlertHW_DF33 = pd.concat(temp2all_5G, axis=0)
        samp_path = './Data/{}/Alert_Deal/Samp_{}_{}/'.format(distr, mode, ftype)
        if os.path.exists(samp_path) == False:
            mkdir(samp_path)
        AlertHW_DF3['告警名称'] = AlertHW_DF3['告警名称'].map(lambda x: x.replace('\/', '/'))
        AlertHW_DF3.to_csv('./Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_5G.csv'.format(distr, mode, ftype, date), index=False,encoding='gbk')
        AlertHW_DF33.to_csv('./Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_delJZ_5G.csv'.format(distr, mode, ftype, date),index=False, encoding='gbk')





