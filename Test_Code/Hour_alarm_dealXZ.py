import os
import json
import pandas as pd
import numpy as np
import csv
import datetime

# alarm_path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据调研\样例数据-告警数据/'
# rule_path = 'E:\PycharmProjects\XZ_Pro\Data\汇总\Project_Opt/告警中英文匹配.xlsx'
# wuxian_save_path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据调研\样例数据-告警数据/'
# dongh_save_path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据调研\样例数据-告警数据/'
#
# alarm_data = pd.read_csv(alarm_path+'历史测试导出新(1).csv',encoding='gbk')
# alarm_rule = pd.read_excel(rule_path)
# columns = alarm_rule['中文'].values.tolist()
#
# alarm_data.columns = columns
# # alarm_data.to_csv(alarm_path+'4new.csv',encoding='gbk',index=False)
# alarm_data['告警发生时间'] = alarm_data['告警发生时间'].map(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d %H:%M:%S'))
#
# hour_range = [hour_list for hour_list in range(0,24)]
#
# date_low = '20220501'
# date_high = '20220502'
# date1 = pd.date_range(date_low, date_high)
# date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()
#
# for alarm_date in date_list:
# # for alarm_date in ['20220605']:
# #     alarm_Data = pd.read_csv(alarm_path+'ALARM-{}.csv'.format(alarm_date),encoding='gbk')
#     # alarm_Data_sel = alarm_Data[['网元类型','告警标题','发生时间','网管告警级别','定位信息','设备厂家','网元名称','三级区域名称','地区','告警工程状态','二级专业','网元ID','设备清除告警时间','机房名称','告警标准名']]
#     # alarm_Data_sel.columns=['设备类型','告警标题','告警发生时间','告警级别','定位信息','设备厂家名称','网元名称','县市','地区','告警工程状态','专业','基站编号','清除时间','设备机房','告警标准名']
#     alarm_data_low = datetime.datetime.strptime(alarm_date,'%Y%m%d')
#     alarm_data_high = datetime.datetime.strptime(alarm_date,'%Y%m%d') + datetime.timedelta(days=1)
#
#     alarm_Data = alarm_data[(alarm_data['告警发生时间']>=alarm_data_low) & (alarm_data['告警发生时间']<alarm_data_high)]
#     alarm_Data['告警小时'] = alarm_Data['告警发生时间'].map(lambda x:datetime.datetime.strftime(x,"%Y%m%d%H"))
#     #
#     hour_day_range = set(alarm_Data['告警小时'])
#
#     for hour_range in hour_day_range:
#         alarm_fin = alarm_Data[alarm_Data['告警小时']==hour_range]
#         alarm_fin.drop(['告警小时'],inplace=True,axis=1)
#         alarm_fin_wuxian = alarm_fin[(alarm_fin['一级专业']=='无线接入网')]
#         alarm_fin_wuxian.to_csv(wuxian_save_path+'wuxian_alarm_{}0000.csv'.format(hour_range),index=False,encoding='utf-8')
#         alarm_fin_dongh = alarm_fin[alarm_fin['一级专业'] == '动环']
#         alarm_fin_dongh.to_csv(dongh_save_path + 'donghuan_alarm_{}0000.csv'.format(hour_range), index=False,
#                                 encoding='utf-8')


# --------------- 活动告警处理 ---------------
alarm_path = '/data/Origin_Alarm_data/'
# alarm_path = 'E:\PycharmProjects\XZ_Pro\Data\全省\Alert_Data\历史数据导出7月/'
# save_path = 'E:\PycharmProjects\XZ_Pro\Data\全省\Alert_Data\历史数据导出7月/'
save_path = '/data/Alarm_deal_data/'

date_low = '20220801'
date_high = '20220831'
date1 = pd.date_range(date_low, date_high)
date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()

active_alarm = pd.read_csv(alarm_path + '202208huodong.csv',encoding='gbk')

active_alarm['EVENT_TIME'] = active_alarm['EVENT_TIME'].map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

for alarm_date in date_list:
    print(alarm_date)
# for alarm_date in ['20220605']:
    alarm_Data = pd.read_csv(alarm_path+'{}.csv'.format(alarm_date),encoding='gbk')
    # alarm_Data_sel = alarm_Data[['网元类型','告警标题','发生时间','网管告警级别','定位信息','设备厂家','网元名称','三级区域名称','地区','告警工程状态','二级专业','网元ID','设备清除告警时间','机房名称','告警标准名']]
    # alarm_Data_sel.columns=['设备类型','告警标题','告警发生时间','告警级别','定位信息','设备厂家名称','网元名称','县市','地区','告警工程状态','专业','基站编号','清除时间','设备机房','告警标准名']

    alarm_data_low = datetime.datetime.strptime(alarm_date,'%Y%m%d')
    alarm_data_high = datetime.datetime.strptime(alarm_date,'%Y%m%d') + datetime.timedelta(days=1)

    alarm_Data_sel = active_alarm[(active_alarm['EVENT_TIME']>=alarm_data_low) & (active_alarm['EVENT_TIME']<alarm_data_high)]
    alarm_Data_fin = pd.concat([alarm_Data,alarm_Data_sel])
    alarm_Data_fin.to_csv(save_path+'ALARM-{}.csv'.format(alarm_date),encoding='gbk',index=False)
