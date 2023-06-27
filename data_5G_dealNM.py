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
# real_warns = ["eNodeB退服告警","基站退服","MME衍生基站退服","小区不可用告警","网元连接中断"] # 4G华为
# real_warns = ["MME衍生基站退服","LTE小区退出服务","MME衍生小区退服","基站退服","网元断链告警"] # 4G中兴
real_warns = ["gNodeB退服告警","NR小区不可用告警","小区不可用告警","网元连接中断"]  # 5G华为兴
# distr_list=[ '宝鸡', '铜川','安康','商洛', '渭南'] # '西安', '咸阳', '延安', '榆林', '汉中',

Data_org11 = pd.read_csv('F:\AIOps_LN\智能基站运维数据-LN/LN_5G_HW.csv',iterator=True,encoding='utf-8')
loop = True
chunkSize = 500000
chunks_4G_HW = []
num = 0
while loop:
    print('已执行{}行数据'.format(num))
    try:
        Data_org = Data_org11.get_chunk(chunkSize)

        alert_dropna = Data_org.dropna(subset=['网元名称', '基站ID'])
        alert_dropna = alert_dropna#[['一级网络类型', '网络类型', '设备类型', '基站名称', 'ENODEB_ID', '所属机房', '设备厂家名称', '设备名称', '地市名称', '告警对象名称', '告警对象设备类型','网管告警级别', '告警标题', '告警发生时间', '告警清除状态', '告警清除时间', '设备工程状态', '告警工程状态']]
        # alert_dropna.rename(columns={'基站名称' :'网元名称'}, inplace=True)

        Data_org = alert_dropna[alert_dropna['网络类型'] == '5G']

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
        # Data_org['告警日期'] = Data_org['告警发生时间'].map(lambda x: x.split()[0])
        # Data_org['告警日期1'] = Data_org['告警日期'].map(
        #     lambda x: datetime(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))

        Data_target = Data_org#[(Data_org['告警日期1'] >= file_date) & (Data_org['告警日期1'] <= file_date)]
        Alert_XJ = Data_target[['告警标题', '告警发生时间', '网元名称', '基站ID', '设备厂家名称', '地市名称', '告警持续时间']]
        Alert_XJ.columns = ['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间']
        Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('地区')[0])

        # 告警标题里 '\/' 转为 '/'
        Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x :x.replace('\/' ,'/'))
        try \
            :Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x :x.replace('\\/' ,'/'))
        except \
            :print(11)

        Data_org = Alert_XJ[['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间']]
        Data_Org1 = Data_org
        Data_org2 = Data_Org1
        Data_org3 = Data_org2[~Data_org2['基站id'].isin(['-',' -',' '])]  # 空白id = '-'

        Data_org3['告警开始时间'] = Data_org3['告警开始时间'].map(lambda x: str(x).replace('-', '/'))


        # 剔除非影响业务告警
        def del_unimp(Data_org):
            Data_org41 = Data_org
            HW_alarmname = Data_org41['告警名称'].values.tolist()
            HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/剔除告警标题.csv', encoding='GBK',engine='python')
            HW_Per_list = np.ravel(HW_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
            # if distr == '渭南':
            #     HW_Per_list = HW_Per_list + ['用户面承载链路故障告警','模块故障']
            HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
            HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['告警名称'])  # 剔除+去重
            Data_org4 = pd.merge(Data_org41, HW_fault_Pd)
            # Data_org4 = Data_org4.sort_values(by='告警开始时间')
            XN_alarmname = Data_org4['告警名称'].values.tolist()
            return Data_org4,XN_alarmname
        # 剔除性能告警
        def del_xn(Data_org4,XN_alarmname):
            XN_Per = pd.read_csv(r'./Data/汇总/Project_Opt/性能告警标题.csv', encoding='GBK',engine='python')
            XN_Per_list = np.ravel(XN_Per.values).tolist()  # series -> 1-D数组（ndarray）-> list  剔除告警标题
            Del_XN_list = list(set(XN_alarmname).difference(set(XN_Per_list)))  # 取补集(set1.difference(ser2)=set1-set2)
            Del_XN_Pd = pd.DataFrame(Del_XN_list, columns=['告警名称'])  # 剔除+去重
            Data_org5 = pd.merge(Data_org4, Del_XN_Pd)
            Data_org5['制式'] = '4G'
            # Data_org5 = Data_org5.sort_values(by='告警开始时间')
            def del_min(x):
                # print(x)
                aa = x.split(':')[0] + ':' + x.split(':')[1]
                return aa
            Data_org5['告警开始分钟'] = Data_org5['告警开始时间'].map(lambda x:del_min(x))
            return Data_org5
        # 剔除告警持续时间小于3分钟的告警
        def del_mins(Data_org5):
            grp_cnt = 1500
            del_alm = []
            del_min3 = ['用户面承载链路故障告警', 'X2接口故障告警','License试运行告警']
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
            if os.path.exists(r'./Data/汇总/Project_Opt/每天退服基站.csv'):
                HW_Per = pd.read_csv(r'./Data/汇总/Project_Opt/每天退服基站.csv', encoding='GBK',engine='python')
                HW_Per_list = np.ravel(HW_Per.values).tolist()
                HW_fault_list = list(set(HW_alarmname).difference(set(HW_Per_list)))  # 取补集
                HW_fault_Pd = pd.DataFrame(HW_fault_list, columns=['网元名称'])
                Data_org10 = pd.merge(Data_org9, HW_fault_Pd, on='网元名称')
                # Data_org11 = Data_org10.sort_values(by='告警开始时间')
            else:
                print('./Data/汇总/Project_Opt/每天退服基站.csv Not Existed!')
                Data_org11 = Data_org9
            return Data_org11
        # 剔除9-11 17-19载波调度站 发生网元连接中断告警 且1分钟内恢复的
        # def del_zb(Data_org11):
        #     Data_org11 = Data_org11.copy()
        #     if os.path.exists(r'./Data/{}/Project_Opt/载波调度清单.xlsx'.format(distr)):
        #         Car_sch = pd.read_excel(r'./Data/{}/Project_Opt/载波调度清单.xlsx'.format(distr))
        #         Data_org11['告警开始小时'] = Data_org11['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
        #         Data_org11['告警持续时间'] = Data_org11['告警持续时间'].map(lambda x: int(x))
        #         Data_org12 = Data_org11[~(Data_org11['网元名称'].isin(Car_sch['基站名称']) & Data_org11['告警开始小时'].isin(
        #             ['09', '10', '11', '17', '18', '19']) & (Data_org11['告警持续时间'] <= 60))]
        #         Data_org13 = Data_org12.copy()
        #         Data_org13['制式'] = '5G'
        #         Data_org13 = Data_org13[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家']]
        #     else:
        #         print(r'./Data/{}/Project_Opt/载波调度清单.xlsx Not Existed!'.format(distr))
        #         Data_org13 = Data_org11
        #     return Data_org13
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
            TF_org = Data_org[Data_org['告警名称'].isin(real_warns)]
            TF_org1 = TF_org.drop_duplicates(subset=['基站id', 'data_hour'])
            tmp = Counter(TF_org1['data_hour'])
            tmp = sorted(tmp.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            try:
                max_hour = tmp[0][0]
            except:
                return Data_org
            tf_id = len(set(TF_org['基站id']))
            if tf_id >= (0.6 * sum_id):
                Data_org1 = Data_org[~((Data_org['data_hour'] == max_hour) & (Data_org['告警名称'].isin(real_warns)))]
                print('{}退服基站数量异常{}/{},{}：00退服基站{},删除告警{}条'.format(date, tf_id, sum_id, max_hour, tmp[0][1],
                                                                   Data_org.shape[0] - Data_org1.shape[0]))
            else:
                Data_org1 = Data_org
            Data_org1.drop(columns=['告警开始时间1', 'date', 'data_hour'])
            return Data_org1


        # Data_org4, XN_alarmname = del_unimp(Data_org3)
        # Data_org5 = del_xn(Data_org4,XN_alarmname)
        # Data_org7 = del_mins(Data_org5)
        # Data_org11 = del_daily(Data_org7)
        # # Data_org13 = del_zb(Data_org11)
        # Data_org14 = del_night(Data_org11)

        tf_alarm = Data_org3['告警名称'].values.tolist()
        Data_tf = del_xn(Data_org3,tf_alarm)
        Data_tf = max_tfwarn(Data_tf)
        Data_tf = del_mins(Data_tf)
        Data_tf = Data_tf[['告警开始时间', '告警持续时间', '告警名称', '基站id', '网元名称', '制式', '厂家','区域（地市）']]
        # if distr in ['铜川','汉中']:
        #     Data_tf = del_daily(Data_tf)
        # Data_tf = del_zb(Data_tf)
        # region_del_path = './Data/{}/Alert_Data/'.format(distr)
        # if os.path.exists(region_del_path) == False:
        #     mkdir(region_del_path)

        # Data_tf.to_csv(r'F:\AIOps_ShanX\智能基站运维数据-LN/故障_LN_4G_HW_delJZ.csv', index=None, encoding='gbk')
        # Data_delJZ = Data_tf
        chunks_4G_HW.append(Data_tf)
        num = num + 500000
    except StopIteration:
        loop = False
        print("Iteration is stopped.")

df_WN = pd.concat(chunks_4G_HW, ignore_index=True)
df_WN = df_WN.sort_values(by='告警开始时间')
df_WN.to_csv(r'F:\AIOps_LN\智能基站运维数据-LN/故障_LN_5G_HW_delJZ.csv', index=None, encoding='gbk')