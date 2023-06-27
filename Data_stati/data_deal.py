# -*- coding: utf-8 -*-
"""
@author: nzl
"""


def data_process_new():
    # 导入库函数
    import numpy as np
    # import matplotlib.pyplot as plt
    from pandas import read_csv
    import pandas as pd
    import os
    import datetime
    import time

    def get_current_week():
        monday, sunday = datetime.date.today(), datetime.date.today()
        one_day = datetime.timedelta(days=1)
        while monday.weekday() != 0:
            monday -= one_day
        while sunday.weekday() != 6:
            sunday += one_day
        monday, sunday = monday - datetime.timedelta(days=7), sunday - datetime.timedelta(days=7)
        return monday, sunday

    def get_double_current_week():
        monday, sunday = datetime.date.today(), datetime.date.today()
        one_day = datetime.timedelta(days=1)
        while monday.weekday() != 0:
            monday -= one_day
        while sunday.weekday() != 6:
            sunday += one_day
        monday, sunday = monday - datetime.timedelta(days=14), sunday - datetime.timedelta(days=7)
        return monday, sunday

    ori_PATH = '/data/resources'
    alarm_path = 'alarm/wuxian/'
    analyse_start_date, analyse_end_date = get_current_week()
    print(analyse_start_date, analyse_end_date)
    analyse_date_list = [str(date).split()[0].replace('-', '') for date in
                         pd.date_range(analyse_start_date, analyse_end_date)]
    print(analyse_date_list)
    date = analyse_date_list[0] + '-' + analyse_date_list[-1][-4:]
    # analyse_start_date_2, analyse_end_date_2 = get_double_current_week()
    # analyse_date_list_2 = [str(date).split()[0].replace('-', '') for date in pd.date_range(analyse_start_date_2, analyse_end_date_2)]

    data = pd.DataFrame()
    abs_path = os.path.join(ori_PATH, alarm_path)
    alarm_filename_list = ['wuxian_alarm_' + filedate + '.csv' for filedate in analyse_date_list]
    for filename in alarm_filename_list:
        try:
            data1 = pd.read_csv(open(os.path.join(abs_path, filename), encoding='utf-8'),
                                names=["设备类型", "告警标题", "告警发生时间", "告警级别", "定位信息",
                                       "设备厂家名称", "网元名称", "县市", "地区", "告警工程状态",
                                       "专业", "基站编号", "清除时间", "设备机房", "机房名称"], engine='python', encoding='utf-8')
        except:
            continue
        print(filename)
        data = pd.concat([data, data1], ignore_index=True)
        print(len(data))
    data.sort_values(by='告警发生时间', inplace=True)
    print(data.head())
    print(data['设备类型'].value_counts())
    data1 = data[data['设备类型'].isin(['EnodeB'])]
    print('4g告警共：', data1.shape)
    print('4g告警共：', data1.shape)
    data2 = data[data['设备类型'].isin(['GNodeB'])]
    data_4g = data1[['告警标题', '告警发生时间', '网元名称', '设备厂家名称', '地区', '清除时间', '告警工程状态', '定位信息']]
    data_4g.columns = ['告警名称', '告警开始时间', '网元名称', '厂家', '区域（地市）', '清除时间', '是否工程', '定位信息']
    data_5g = data2[['告警标题', '告警发生时间', '网元名称', '设备厂家名称', '地区', '清除时间', '告警工程状态', '定位信息']]
    data_5g.columns = ['告警名称', '告警开始时间', '网元名称', '厂家', '区域（地市）', '清除时间', '是否工程', '定位信息']

    print(data_4g.isna().sum())
    print(data_5g.isna().sum())
    data_4g.dropna(subset=['网元名称', '告警名称', '厂家', '区域（地市）', '是否工程'], inplace=True)
    data_5g.dropna(subset=['网元名称', '告警名称', '厂家', '区域（地市）', '是否工程'], inplace=True)
    print(data_4g.isna().sum())
    print(data_5g.isna().sum())
    data_4g['告警开始时间'] = pd.to_datetime(data_4g['告警开始时间'])
    data_4g['告警开始时间'] = data_4g['告警开始时间'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    data_5g['告警开始时间'] = pd.to_datetime(data_5g['告警开始时间'])
    data_5g['告警开始时间'] = data_5g['告警开始时间'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    print('剔除工程态前：', data_4g.shape)
    data_4g = data_4g[data_4g['是否工程'] != '1500']  # 剔除工程告警1200,1500,9900
    data_4g = data_4g[data_4g['是否工程'] != '1200']  # 剔除工程告警1200,1500,9900
    data_4g = data_4g[data_4g['是否工程'] != '9900']  # 剔除工程告警1200,1500,9900
    print('剔除工程态后：', data_4g.shape)
    data_5g = data_5g[data_5g['是否工程'] != '1500']
    data_5g = data_5g[data_5g['是否工程'] != '1200']
    data_5g = data_5g[data_5g['是否工程'] != '9900']
    data_4g.sort_values(by=['告警开始时间'], inplace=True)
    data_5g.sort_values(by=['告警开始时间'], inplace=True)
    data_4g_out = data_4g[['告警开始时间', '告警名称', '网元名称', '厂家', '区域（地市）', '清除时间']]
    data_5g_out = data_5g[['告警开始时间', '告警名称', '网元名称', '厂家', '区域（地市）', '清除时间']]
    data_4g_out = data_4g_out[pd.to_datetime(data_4g_out['告警开始时间']) >= pd.to_datetime(analyse_start_date)]
    data_5g_out = data_5g_out[pd.to_datetime(data_5g_out['告警开始时间']) >= pd.to_datetime(analyse_start_date)]
    # 基站告警统计用数据
    if not os.path.exists('/data/alarm' + '/' + date + '/enodeb'):
        os.makedirs('/data/alarm' + '/' + date + '/enodeb')
    data_4g_out.to_csv('/data/alarm' + '/' + date + '/enodeb/' + 'ZJ_wuxian_alarm_4G.csv', header=True, index=False,
                       encoding='utf-8')
    if not os.path.exists('/data/alarm' + '/' + date + '/gnodeb'):
        os.makedirs('/data/alarm' + '/' + date + '/gnodeb')
    data_5g_out.to_csv('/data/alarm' + '/' + date + '/gnodeb/' + 'ZJ_wuxian_alarm_5G.csv', header=True, index=False,
                       encoding='utf-8')
    # data_5g_out.to_csv(ori_PATH+'/'+date+'/'+'ZJ_wuxian_alarm_5G.csv', header=True, index=False, encoding='gbk')

    ## 处理工单数据
    data_order = pd.DataFrame()
    order_path = 'order/'
    abs_order_path = os.path.join(ori_PATH, order_path)
    order_filename_list = ['order_' + filedate + '.csv' for filedate in analyse_date_list]
    for filename in order_filename_list:
        try:
            data1 = pd.read_csv(open(os.path.join(abs_order_path, filename), encoding='utf-8'),
                                names=["工单类型", "工单号", "发生地区", "故障发生时间", "工单报结时间", "工单录入时间",
                                       "工单标题", "网络分类", "故障设备类型", "故障设备厂商", "网元名称", "历时",
                                       "工单状态", "故障原因分类", "处理过程", "故障恢复时间", "申请报结时间",
                                       "申请报结人", "基站号", "基站名", "告警清除时间"], engine='python', encoding='utf-8')
        except:
            continue
        print(filename)
        data_order = pd.concat([data_order, data1], ignore_index=True)
        print(len(data_order))
    print(data_order.head())
    print(data_order.columns.values.tolist())
    data_order = data_order[['故障设备类型', '故障设备厂商', '工单号', '发生地区', '故障发生时间', '工单标题',
                             '历时', '网元名称', '工单状态', '故障恢复时间',
                             '申请报结人', '工单报结时间', '故障原因分类']]
    data_order.columns = ['故障设备类型', '故障设备厂商', '工单号', '发生地区', '发生时间', '工单标题', '历时', '网元名称', '工单状态', '故障消除时间', '申请报结人', '申请报结班组',
                          '故障原因类别']
    data_order.to_csv('/data/wo' + '/' + 'wo_{}.csv'.format(date), header=True, index=False, encoding='utf-8')

    # 处理资源工参数据
    nrm_path = '/data/ftp/data/resources/nrm/'
    # abs_nrm_path = os.path.join(ori_PATH, nrm_path)
    abs_nrm_path = nrm_path
    #####4G
    # order_filename_list = ['4G_enodeb_' + filedate + '.csv' for filedate in analyse_date_list]
    order_filename_list = ['mod-11546299-143135151_DW_DM_ZY_ENODEB_' + filedate + '000000' + '.csv' for filedate in
                           analyse_date_list]
    # data_nrm = pd.read_csv(open(os.path.join(abs_nrm_path, order_filename_list[-5]), encoding='gbk'), delimiter='\$\$',
    #                            names=["E-NODEB名称", "E-NODEBID", "网管中网元名称", "所属机房/位置点", "覆盖类型",
    #                                   "VIP级别", "所属省", "所属地市", "所属区县", "设备厂家", "生命周期状态",
    #                                   "维护分公司", "规划站号", "经度", "维度"], encoding='utf-8')
    data_nrm = pd.read_csv(open(os.path.join(abs_nrm_path, order_filename_list[-5]), encoding='utf-8'), delimiter='\$\$',
                           names=['NODEB_ID', 'USERLABEL', 'RELATED_ROOM_LOCATION', 'LIFECYCLE_STATUS',
                                  'BEEHIVE_TYPE', 'VIP_TYPE', 'VENDOR_ID', 'ZH_LABEL', 'PROV_NAME',
                                  'AREA_NAME', 'CITY_NAME'], encoding='utf-8')
    # data_nrm = data_nrm[
    #     ["规划站号", "E-NODEB名称", "网管中网元名称", "生命周期状态", "设备厂家", "VIP级别", "E-NODEBID", "所属省", "所属地市", "所属区县", "所属机房/位置点",
    #      "维护分公司", "覆盖类型"]]
    # data_nrm.columns = ["基础小区号", "基站名称", "网管中网元名称", "生命周期状态", "生产厂家", "VIP级别", "ENODEBID", "所属省", "所属地市", "所属区县",
    #                     "所属机房", "所属区县分公司", "覆盖类型"]
    data_nrm = data_nrm[['NODEB_ID', 'AREA_NAME']]
    data_nrm.columns = ['NODEB_ID', '所属地市']
    data_nrm.to_csv('/data/nrm/nrm_4G.csv', header=True, index=False, encoding='utf-8')

    #####5G
    # order_filename_list = ['5G_gnodeb_' + filedate + '.csv' for filedate in analyse_date_list]
    order_filename_list = ['mod-11546299-85876849_DW_DM_ZY_GNODEB_' + filedate + '000000' + '.csv' for filedate in
                           analyse_date_list]
    # data_nrm = pd.read_csv(open(os.path.join(abs_nrm_path, order_filename_list[-5]), encoding='gbk'), delimiter='\$\$',
    #                            names=["G-NODEB名称", "G-NODEB_ID", "网管中网元名称", "所属机房/位置点",
    #                                   "覆盖类型", "VIP级别", "所属省", "所属地市", "所属区县", "设备厂家",
    #                                   "生命周期状态", "维护分公司", "规划站号", "经度", "维度"], encoding='utf-8')
    data_nrm = pd.read_csv(open(os.path.join(abs_nrm_path, order_filename_list[-5]), encoding='utf-8'), delimiter='\$\$',
                           names=['NODEB_ID', 'USERLABEL', 'RELATED_ROOM_LOCATION', 'LIFECYCLE_STATUS',
                                  'BEEHIVE_TYPE', 'VIP_TYPE', 'VENDOR_ID', 'ZH_LABEL', 'PROV_NAME',
                                  'AREA_NAME', 'CITY_NAME'], encoding='utf-8')
    # data_nrm = data_nrm[["网管中网元名称", "生命周期状态", "设备厂家", "VIP级别", "G-NODEB_ID", "所属机房/位置点", "覆盖类型", "G-NODEB名称", "所属地市"]]
    # data_nrm.columns = ["基站名称", "生命周期状态", "生产厂家", "VIP级别", "ENODEBID", "所属机房", "覆盖类型", "网络制式", "所属地市"]
    data_nrm = data_nrm[['NODEB_ID', 'AREA_NAME']]
    data_nrm.columns = ['NODEB_ID', '所属地市']
    data_nrm.to_csv('/data/nrm/nrm_5G.csv', header=True, index=False, encoding='utf-8')


if __name__ == '__main__':
    data_process_new()
