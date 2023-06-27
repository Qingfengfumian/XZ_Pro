import time,os
import pandas as pd
import numpy as np
from datetime import datetime
import time,os
from datetime import datetime
import warnings
from collections import Counter

def mkdir(path):
    path = path.strip()
    path = path.rstrip("//") # 删除 string 字符串末尾的指定字符
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
real_warns_5G = ["eNodeB退服告警","gNodeB退服告警","NR小区不可用告警","射频单元维护链路异常告警","小区不可用告警","网元连接中断"] # 铜川
date_low = '20210813'
date_high = '20210829'
date1 = pd.date_range(date_low, date_high)
date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()
for date_time in date_list:
    distr_list=['宝鸡', '铜川','安康','商洛', '渭南', '西安', '咸阳', '延安', '榆林', '汉中']
    for i_region in distr_list:
        distr = i_region
        print('{}-{}'.format(i_region,date_time))
        try:Data_org = pd.read_csv('E:\AItask\AIOps_ShanX\服务器数据\告警\告警日志_5G/{}/告警日志{}_5G.csv'.format(i_region,date_time),encoding='gbk')
        except:continue
        Data_org = Data_org[['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '告警持续时间']]
        # Data_org44 = Data_org[Data_org['厂家']=='华为']
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
            HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/剔除告警标题.csv', encoding='GBK', engine='python')
            HW_Per_list = np.ravel(HW_Per['alarmname'].values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
            HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
            HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['告警名称'])  # 剔除+去重
            Data_org4 = pd.merge(Data_org41, HW_fault_Pd)
            Data_org4 = Data_org4.sort_values(by='告警开始时间')
            XN_alarmname = Data_org4['告警名称'].values.tolist()
            return Data_org4, XN_alarmname

        # 剔除性能告警
        def del_xn(Data_org4, XN_alarmname):
            XN_Per = pd.read_csv(r'./Data/汇总/Project_Opt/性能告警标题.csv', encoding='GBK', engine='python')
            XN_Per_list = np.ravel(XN_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
            Del_XN_list = list(set(XN_alarmname).difference(set(XN_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
            Del_XN_Pd = pd.DataFrame(Del_XN_list, columns=['告警名称'])  # 剔除+去重
            Data_org5 = pd.merge(Data_org4, Del_XN_Pd)
            Data_org5 = Data_org5.sort_values(by='告警开始时间')
            Data_org5['告警开始分钟'] = Data_org5['告警开始时间'].map(lambda x: x.split(':')[0] + ':' + x.split(':')[1])
            return Data_org5

        del_alm = []
        grp_cnt = 1500
        del_min3 = ['用户面承载链路故障告警','时钟参考源异常告警','小区服务能力下降告警', 'X2接口故障告警','License试运行告警']  # 铜川 = 汉中
        # 剔除告警持续时间小于3分钟的告警
        def del_mins(Data_org5):
            Group_mini = Data_org5.groupby('告警开始分钟').count()
            Group_count = Group_mini[Group_mini['告警开始时间'] > grp_cnt]
            del_mini = Group_count.index.tolist()
            Data_org6 = Data_org5
            for i in del_mini:
                if del_alm:
                    del_lis = '|'.join(del_alm)
                    Data_org6 = Data_org6[
                        ~(Data_org6['告警开始时间'].str.contains(i) & (Data_org6['告警名称'].str.contains(del_lis)))]
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
                HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/每天退服基站.csv'.format(distr), encoding='GBK', engine='python')
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
                Data_org1 = Data_org[~((Data_org['data_hour'] == max_hour) & (Data_org['告警名称'].isin(real_warns_5G)))]
                print('{}退服基站数量异常{}/{},{}：00退服基站{},删除告警{}条'.format(date, tf_id, sum_id, max_hour, tmp[0][1],
                                                                   Data_org.shape[0] - Data_org1.shape[0]))
            else:
                Data_org1 = Data_org
            Data_org1.drop(columns=['告警开始时间1', 'date', 'data_hour'])
            return Data_org1


        Data_org4, XN_alarmname = del_unimp(Data_org3)
        Data_org5 = del_xn(Data_org4, XN_alarmname)
        Data_org7 = del_mins(Data_org5)
        Data_org11 = del_daily(Data_org7)
        Data_org13 = del_zb(Data_org11)
        Data_org14 = del_night(Data_org13)

        tf_alarm = Data_org3['告警名称'].values.tolist()
        Data_tf = del_xn(Data_org3, tf_alarm)
        Data_tf = max_tfwarn(Data_tf)
        Data_tf = del_mins(Data_tf)
        # if distr in ['铜川','汉中']:
        #     Data_tf = del_daily(Data_tf)
        Data_tf = del_zb(Data_tf)
        region_del_path = './Data/{}/Alert_Data/'.format(distr)
        if os.path.exists(region_del_path) == False:
            mkdir(region_del_path)

        Data_tf.to_csv(r'E:\PycharmProjects\BS_Monitor_V1\Data/{}/Alert_Data/故障_{}_delJZ_5G.csv'.format(distr, date_time), index=None, encoding='gbk')
        Data_delJZ = Data_tf

        Data_org14.to_csv(r'E:\PycharmProjects\BS_Monitor_V1\Data/{}/Alert_Data/故障_{}_5G.csv'.format(distr, date_time), index=None, encoding='gbk')
        Data = Data_org14

