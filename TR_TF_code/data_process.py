import time,os
import pandas as pd
import numpy as np
import threading
import re
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


real_warns = ["MME衍生基站退服","LTE小区退出服务","MME衍生小区退服","基站退服","网元断链告警"]


def split_distr(alert_data):
    # alert_data.columns = ['一级网络类型', '网络类型', '设备类型', '网元名称', '所属机房', '设备厂家名称', '设备名称', '地市名称', '告警对象名称',
    #                       '告警对象设备类型', '网管告警级别', '告警标题', '告警发生时间', '告警清除状态', '告警清除时间', '设备工程状态', '告警工程状态']
    alert_4G = alert_data[alert_data['网络类型'] == 103]

    # opt_select = opt_data[['基站名称', 'ENODEB_ID']]
    # opt_select.drop_duplicates(subset=['ENODEB_ID'], keep='first', inplace=True)
    #
    # alert_merge = pd.merge(alert_4G, opt_select, left_on='网元名称', right_on='ENODEB_ID', how='left')
    # alert_del = alert_merge.drop(columns=['ENODEB_ID'])

    alert_dropna = alert_4G.dropna(subset=['基站名称', 'ENODEB_ID'])
    alert_dropna = alert_dropna[
        ['一级网络类型', '网络类型', '设备类型', '基站名称', 'ENODEB_ID', '所属机房', '设备厂家名称', '设备名称', '地市名称', '告警对象名称', '告警对象设备类型',
         '网管告警级别', '告警标题', '告警发生时间', '告警清除状态', '告警清除时间', '设备工程状态', '告警工程状态','所属区县']]
   
    alert_dropna.rename(columns={'基站名称':'网元名称'}, inplace=True)
    ##
#     alert_dropna1 = alert_dropna[(alert_dropna['告警工程状态']!=1500) & (alert_dropna['告警工程状态']!=1200)]
#     print('删除工程态告警：%s条'%(alert_dropna.shape[0]-alert_dropna1.shape[0]))
    ##
    Data_org = alert_dropna.copy()
    Data_org['告警持续时间1'] = Data_org['告警发生时间'].map(
        lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
    Data_org['告警清除时间'] = Data_org['告警清除时间'].fillna(-1)
    Data_org['告警持续时间2'] = Data_org['告警清除时间'].map(
        lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))) if x != -1 else -1)
    Data_org['告警持续时间'] = Data_org['告警持续时间2'] - Data_org['告警持续时间1']
    Data_org['告警持续时间'] = Data_org['告警持续时间'].map(lambda x: x if x >= 0 else 9999)
    Data_org = Data_org.drop(['告警持续时间1', '告警持续时间2'], axis=1)

    Data_org['告警日期'] = Data_org['告警发生时间'].map(lambda x: x.split()[0])
    Data_org['告警日期1'] = Data_org['告警日期'].map(
        lambda x: datetime(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))
    
    
#     file_date = pd.to_datetime('2021-1-1 00:00:00', "%Y-%m-%d %H:%M:%S")
#     file_date = '20210101'
#     Data_target = Data_org[Data_org['告警日期1'] < file_date]
    Data_target = Data_org
    Alert_XJ = Data_target[['告警标题', '告警发生时间', '网元名称', 'ENODEB_ID', '设备厂家名称', '地市名称', '告警持续时间','所属区县', '所属机房']]
    Alert_XJ.columns = ['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间','所属区县', '所属机房']
    Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('地区')[0])
    Alert_XJ = Alert_XJ[Alert_XJ['厂家']=='华为']
#     Alert_XJ.to_csv('./data/train_data.csv', index=False,encoding='gbk')
    return Alert_XJ

def data_del_part(alert_data_part):
    """
    :param grp_cnt: 每分钟告警数量异常阈值 例如：1500
    :param del_alm: 每分钟告警数量异常，删除告警类型 例如：['用户面承载链路故障告警','时钟参考源异常告警']
    :param del_min3: 删除告警持续时间小于3分钟的告警  例如：同上
    :param sloc: 读入数据时，用于定位的字段 例如：'级别'
    :param fatr: 数据所属厂家 例如：'华为'
    :return: Data_delJZ：未删除夜间告警
            Data：删除夜间告警
    """
    grp_cnt = 1500
    del_alm = []
    del_min3 =  ['用户面承载链路故障告警', '时钟参考源异常告警', '小区服务能力下降告警', 'X2接口故障告警', 'License试运行告警']
    Data_org = alert_data_part

    Data_org = Data_org[['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '告警持续时间']]#,'所属区县', '所属机房']]
    Data_Org1 = Data_org

    Data_org2 = Data_Org1
    Data_org3 = Data_org2[~Data_org2['基站id'].isin(['-',' -',' '])]  # 空白id = '-'

    Data_org3['告警开始时间'] = Data_org3['告警开始时间'].map(lambda x: str(x).replace('-', '/'))
    Data_org3 = Data_org3.sort_values(by='告警开始时间')
    
# -------------------------- 删除异常时间点的告警 ------------------------
    # 剔除非影响业务告警
    Data_org4 = Data_org3
#     HW_alarmname = Data_org4['告警名称'].values.tolist()
#     HW_Per = pd.read_csv(r'./data/Opt/剔除告警标题v2.csv', encoding='GBK',engine='python')
#     HW_Per_list = np.ravel(HW_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
#     HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
#     HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['告警名称'])  # 剔除+去重
#     Data_org41 = pd.merge(Data_org4, HW_fault_Pd)
#     Data_org41 = Data_org41.sort_values(by='告警开始时间')

    XN_alarmname = Data_org4['告警名称'].values.tolist()

    XN_Per = pd.read_csv(r'./data/Opt/性能告警标题v2.csv', encoding='GBK',engine='python')
    XN_Per_list = np.ravel(XN_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
    Del_XN_list = list(set(XN_alarmname).difference(set(XN_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
    Del_XN_Pd = pd.DataFrame(Del_XN_list, columns=['告警名称'])  # 剔除+去重
    Data_org5 = pd.merge(Data_org4, Del_XN_Pd)
    Data_org5 = Data_org5.sort_values(by='告警开始时间')

    Data_org5['告警开始分钟'] = Data_org5['告警开始时间'].map(lambda x: x.split(':')[0] + ':' + x.split(':')[1])
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
    # 剔除告警持续时间小于3分钟的告警
    Data_org7 = Data_org6[~((Data_org6['告警持续时间'] <= 180) & (Data_org6['告警名称'].isin(del_min3)))]
    Data_org8 = Data_org7.copy()
    Data_org8['制式'] = '4G'
    Data_org8['告警结束时间'] = 0
    Data_org8.drop(['告警持续时间'], axis = 1, inplace = True)
    Data_org8 = Data_org8[['告警开始时间', '告警结束时间', '告警名称', '基站id', '网元名称']]#, '制式', '厂家','所属区县', '所属机房']]
#
#     Data_org9 = Data_org7.copy()
#     HW_alarmname = Data_org9['网元名称'].values.tolist()
#     try:
#         HW_Per = pd.read_csv(r'./data/Opt/宝鸡每日退服清单.csv', encoding='GBK',engine='python')
#     except:
#         HW_Per = pd.read_csv(r'./data/Opt/宝鸡每日退服清单.csv', encoding='utf-8',engine='python')
        
#     HW_Per_list = np.ravel(HW_Per.values).tolist()
#     HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集
#     HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['网元名称'])
#     Data_org10 = pd.merge(Data_org9, HW_fault_Pd, on='网元名称')
#     Data_org11 = Data_org10.sort_values(by='告警开始时间')

    # 剔除9-11 17-19载波调度站 发生网元连接中断告警 且1分钟内恢复的
#     Data_org11 = Data_org11.copy()
#     Car_sch = pd.read_excel(r'./Data/{}/Project_Opt/载波调度清单.xlsx'.format(distr))
#     Data_org11['告警开始小时'] = Data_org11['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
#     Data_org11['告警持续时间'] = Data_org11['告警持续时间'].map(lambda x: int(x))
#     Data_org12 = Data_org11[~(Data_org11['网元名称'].isin(Car_sch['基站名称']) & Data_org11['告警开始小时'].isin(
#         ['09', '10', '11', '17', '18', '19']) & (Data_org11['告警持续时间'] <= 60))]
#     Data_org13 = Data_org12.copy()
#     Data_org13['制式'] = '4G'
#    Data_org13['告警结束时间'] = 0
#    Data_org13.drop(['告警持续时间'], axis=1, inplace=True)
    
    Data_org13 = Data_org7[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称']]#, '厂家','所属区县', '所属机房']]
#     region_del_path = './Data/{}/Alert_Data/'.format(distr)
#     if os.path.exists(region_del_path) == False:
#         mkdir(region_del_path)
#     Data_org13.to_csv(r'./data/train_delJZ_NoList.csv', index=None,encoding='gbk')
#     Data_delJZ = Data_org13
#     # 剔除夜间告警
#     Data_org14 = Data_org13
#     Data_org14['告警开始小时'] = Data_org14['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
#     Data_org14 = Data_org14[~Data_org14['告警开始小时'].isin(['23', '00', '01', '02', '03', '04', '05', '06'])]
#     Data_org14 = Data_org14.drop(['告警开始小时'], axis=1)
#     Data_org14.to_csv(r'./Data/{}/Alert_Data/故障_{}.csv'.format(distr,date), index=None,encoding='gbk')
#     Data = Data_org14
    def max_tfwarn(Data_org):
        Data_org['告警开始时间1'] = pd.to_datetime(Data_org['告警开始时间'])
        Data_org['date'] = pd.to_datetime(Data_org['告警开始时间1'].dt.date)
        date_list = list(set(Data_org['date'].to_list()))
        date_list = sorted(date_list,reverse=False)
        Data_org3_l = []
        for date in date_list:
            Data_org1 = Data_org[Data_org['date']==date]
            sum_id = len(set(Data_org1['基站id']))
            Data_org1["data_hour"] = pd.to_datetime(Data_org1['告警开始时间']).dt.hour
            date = Data_org1.iloc[0]['date']
            TF_org = Data_org1[Data_org1['告警名称'].isin(real_warns)]
            TF_org1 = TF_org.drop_duplicates(subset=['基站id','data_hour'])
            tmp = Counter(TF_org1['data_hour'])
            tmp = sorted(tmp.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
            max_hour = tmp[0][0]
            tf_id = len(set(TF_org['基站id']))
            if tf_id>=(0.6*sum_id):
                Data_org2 = Data_org1[~((Data_org1['data_hour']==max_hour)&(Data_org1['告警名称'].isin(real_warns)))]
                print('{}退服基站数量异常{}/{},{}：00退服基站{},删除告警{}条'.format(date,tf_id,sum_id,max_hour,tmp[0][1],Data_org1.shape[0]-Data_org2.shape[0]))
            else:
                Data_org2 = Data_org1
            Data_org2.drop(columns=['告警开始时间1','date','data_hour'])
            Data_org3_l.append(Data_org2)
        Data_org3 = pd.concat(Data_org3_l)
        return Data_org3
    Data_org14 = max_tfwarn(Data_org13)
    return Data_org14

# train
# try:
#     tmp = pd.read_csv('./data/origin-train/宝鸡2020new.csv',encoding='utf-8')
# except:
#     tmp = pd.read_csv('./data/origin-train/宝鸡2020new.csv',encoding='gbk')
# train_data = split_distr(tmp)
# train_data1 = data_del_part(train_data)
# train_data1.to_csv(r'./data/train_NoList_ALL.csv', index=None,encoding='gbk')

# # test 
# def load_data(file_path):
#     file_list = []
#     for file in os.listdir(file_path):
#         if file.endswith('.csv'):
#             try:
#                 tmp1 = pd.read_csv(os.path.join(file_path,file),encoding='utf-8')
#             except:
#                 tmp1 = pd.read_csv(os.path.join(file_path,file),encoding='gbk')
#             train_data_t = split_distr(tmp1)
#             train_data1_t = data_del_part(train_data_t)

#             file_list.append(train_data1_t)
#         tmp = pd.concat(file_list)
#     return tmp

# train_data11 = load_data('./data/origin-train')
# train_data11.to_csv(r'./data/20200701-20201231_NoList_ALL.csv', index=None,encoding='gbk')

def load_data1(file_path,date_low,date_high,Factory_C,mode_name):
    date1 = pd.date_range(date_low, date_high)
    date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()

    file_list = []
    for date_time in date_list:
        file = '故障_处理_{}_delJZ_{}.csv'.format(date_time,mode_name)
        print(file)
        try:
            try:
                tmp1 = pd.read_csv(os.path.join(file_path,file),encoding='utf-8')
            except:
                tmp1 = pd.read_csv(os.path.join(file_path,file),encoding='gbk')
        except:
            print('故障_处理_{}_delJZ_{}.csv'.format(date_time,mode_name))
            continue
        file_list.append(tmp1)
    tmp = pd.concat(file_list)
    return tmp


def data_process_run(Org_path,Factory_C,mode_name,date_low_train,date_high_train,date_low_pre,date_high_pre):
    # Org_path = 'D:/Pycharm/LN-4G-ZX/SY_4G_ZX/'
    # Factory_C = 'ZX'
    # date_low_train = '20210301'
    # date_high_train = '20210731'
    # date_low_pre = '20210725'
    # date_high_pre = '20210816'

    train_data22 = load_data1('{}/Data/'.format(Org_path),date_low_train,date_high_train,Factory_C,mode_name)
    train_data22.to_csv(r'{}/{}-{}_NoList_ALL.csv'.format(Org_path,date_low_train,date_high_train), index=None,encoding='gbk')

    train_data22 = load_data1('{}/Data/'.format(Org_path),date_low_pre,date_high_pre,Factory_C,mode_name)
    train_data22.to_csv(r'{}/{}-{}_NoList_ALL.csv'.format(Org_path,date_low_pre,date_high_pre), index=None,encoding='gbk')

