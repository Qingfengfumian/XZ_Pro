import time,os
import pandas as pd
import numpy as np
from datetime import datetime
import time,os
from datetime import datetime
import warnings
from collections import Counter

Data_org11 = pd.read_csv('F:\AIOps_ShanX\智能基站运维数据-LN/LN_4G_HW.csv',iterator=True,encoding='utf-8')
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

        Data_org = alert_dropna[alert_dropna['网络类型'] == '4G']

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
        Alert_XJ = Data_target[['告警标题', '告警发生时间', '网元名称', '基站ID', '设备厂家名称', '地市名称', '告警持续时间','网络类型']]
        Alert_XJ.columns = ['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间','网络类型']
        Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('地区')[0])

        # 告警标题里 '\/' 转为 '/'
        Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x :x.replace('\/' ,'/'))
        try \
            :Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x :x.replace('\\/' ,'/'))
        except \
            :print(11)


        chunks_4G_HW.append(Alert_XJ)
        num = num + 500000
    except StopIteration:
        loop = False
        print("Iteration is stopped.")

df_WN = pd.concat(chunks_4G_HW, ignore_index=True)
df_WN = df_WN.sort_values(by='告警开始时间')
df_WN.to_csv(r'F:\AIOps_ShanX\智能基站运维数据-LN/告警日志_LN_4G_HW.csv', index=None, encoding='gbk')