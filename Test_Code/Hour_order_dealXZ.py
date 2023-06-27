import os
import json
import pandas as pd
import numpy as np
import csv
import datetime

# order_path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据\故障工单/'
# rule_path = 'E:\PycharmProjects\XZ_Pro\Data\汇总\Project_Opt/工单中英文匹配.xlsx'
# order_save_path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据\故障工单/'

order_path = '/data/Origin_Order_data/'
rule_path = '/home/XZ_Pro_AIOps/Data/汇总/Project_Opt/工单中英文匹配.xlsx'
order_save_path = '/data/Order_data_hour/'

order_data = pd.read_csv(order_path+'1.csv',encoding='gbk')
order_rule = pd.read_excel(rule_path)
columns = order_rule['中文'].values.tolist()

order_data.columns = columns

order_data.to_csv(order_save_path+'中间工单.csv',index=False,encoding='gbk')
order_data = pd.read_csv(order_save_path+'中间工单.csv',encoding='gbk')
# order_data.rename(columns = {'工单创建时间':'AAAA'},inplace=True)
# order_data.to_csv(order_path+'4new.csv',encoding='gbk',index=False)
order_data['工单创建时间'] = order_data['工单创建时间'].map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

hour_range = [hour_list for hour_list in range(0,24)]

date_low = '20220101'
date_high = '20220131'
date1 = pd.date_range(date_low, date_high)
date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()

for order_date in date_list:
# for alarm_date in ['20220605']:
#     alarm_Data = pd.read_csv(alarm_path+'ALARM-{}.csv'.format(alarm_date),encoding='gbk')
    # alarm_Data_sel = alarm_Data[['网元类型','告警标题','发生时间','网管告警级别','定位信息','设备厂家','网元名称','三级区域名称','地区','告警工程状态','二级专业','网元ID','设备清除告警时间','机房名称','告警标准名']]
    # alarm_Data_sel.columns=['设备类型','告警标题','告警发生时间','告警级别','定位信息','设备厂家名称','网元名称','县市','地区','告警工程状态','专业','基站编号','清除时间','设备机房','告警标准名']
    order_data_low = datetime.datetime.strptime(order_date,'%Y%m%d')
    order_data_high = datetime.datetime.strptime(order_date,'%Y%m%d') + datetime.timedelta(days=1)

    order_Data = order_data[(order_data['工单创建时间']>=order_data_low) & (order_data['工单创建时间']<order_data_high)]
    order_Data['工单小时'] = order_Data['工单创建时间'].map(lambda x:datetime.datetime.strftime(x,"%Y%m%d%H"))
    #
    hour_day_range = set(order_Data['工单小时'])

    for hour_range_1 in hour_day_range:
        alarm_fin = order_Data[order_Data['工单小时']==hour_range_1]
        alarm_fin.drop(['工单小时'],inplace=True,axis=1)
        alarm_fin_wuxian = alarm_fin[alarm_fin['故障专业']=='无线接入网']
        # alarm_fin_wuxian = alarm_fin[(alarm_fin['故障专业']=='无线接入网')|(alarm_fin['故障专业']=='动力环境')]

        # --------------- 20220804 工单字段处理 ----------------
        alarm_fin_wuxian['工单类型'] = '无线'
        alarm_fin_wuxian['申请报结人'] = ' '
        alarm_fin_wuxian['基站号'] = ' '
        alarm_fin_wuxian.drop(columns=['故障设备类型'],inplace=True)

        alarm_fin_wuxian.rename(columns = {'工单编号':'工单号','故障地市':'发生地区','故障开始时间':'故障发生时间','归档时间':'工单报结时间','工单创建时间':'工单录入时间','工单主题':'工单标题','告警逻辑分类':'故障设备类型','故障历时时间':'历时','故障原因细分':'故障原因分类','处理措施':'处理过程','业务恢复时间':'故障恢复时间','T2最终回复时间':'申请报结时间','设备名称':'基站名'},inplace=True)
        order_fin_wuxian = alarm_fin_wuxian[['工单类型','工单号','发生地区','故障发生时间','工单报结时间','工单录入时间','工单标题','网络分类','故障设备类型','故障设备厂商','网元名称','历时','工单状态','故障原因分类','处理过程','故障恢复时间','申请报结时间','申请报结人','基站号','基站名','告警清除时间']]

        order_fin_wuxian.to_csv(order_save_path+'order_{}0000.csv'.format(hour_range_1),index=False,encoding='gbk')
        # alarm_fin_dongh = alarm_fin[alarm_fin['专业'] == '动环']
        # alarm_fin_dongh.to_csv(dongh_path + 'donghuan_alarm_{}0000.csv'.format(hour_range), index=False,
        #                         encoding='utf-8')