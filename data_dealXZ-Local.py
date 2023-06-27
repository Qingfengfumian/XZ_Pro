import pandas as pd
import time


# # ------------- 读原始告警数据 分块处理 -------------
# i = 'F:\AIOps_ShanX\智能基站运维数据-LN/'
# start = time.time()
# yy = pd.read_csv('{}/告警表：ALARM_RAW_ALARMS.csv'.format(i), iterator=True, encoding='utf-8')
# # df = yy.get_chunk(200000)
#
# loop = True
# chunkSize = 1000000
# chunks_4G_HW = []
# chunks_4G_ALX = []
# chunks_4G_ZX = []
# chunks_5G_ZX = []
# chunks_5G_HW = []
# chunks_4G_DT = []
# chunks_4G_NJY = []
# chunks_5G_ALX = []
# num = 0
# while loop:
#     print('已执行{}行数据'.format(num))
#     try:
#         linshidata = yy.get_chunk(chunkSize)
#         linshidata = linshidata[['告警标题','告警发生时间','告警清除时间','网元名称','基站ID','设备厂家名称','地市名称','网络类型']]
#         linshidatab = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '华为')]
#         chunks_4G_HW.append(linshidatab)
#         linshidatac = linshidata[(linshidata['网络类型'] == '4G')  & (linshidata['设备厂家名称'] == '爱立信')]
#         chunks_4G_ALX.append(linshidatac)
#         linshidatad = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '中兴')]
#         chunks_4G_ZX.append(linshidatad)
#         linshidatae = linshidata[(linshidata['网络类型'] == '5G') & (linshidata['设备厂家名称'] == '中兴')]
#         chunks_5G_ZX.append(linshidatae)
#         linshidatae = linshidata[(linshidata['网络类型'] == '5G') & (linshidata['设备厂家名称'] == '华为')]
#         chunks_5G_HW.append(linshidatae)
#         linshidatae = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '大唐')]
#         chunks_4G_DT.append(linshidatae)
#         linshidatae = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '诺基亚')]
#         chunks_4G_NJY.append(linshidatae)
#         linshidatae = linshidata[(linshidata['网络类型'] == '5G') & (linshidata['设备厂家名称'] == '爱立信')]
#         chunks_5G_ALX.append(linshidatae)
#         num = num + 1000000
#     except StopIteration:
#         loop = False
#         print("Iteration is stopped.")
#
# df_WN = pd.concat(chunks_4G_HW, ignore_index=True)
# df_WN.to_csv('{}/LN_4G_HW.csv'.format(i),index=False)
# df_AK = pd.concat(chunks_4G_ALX, ignore_index=True)
# df_AK.to_csv('{}/LN_4G_ALX.csv'.format(i),index=False)
# df_SL = pd.concat(chunks_4G_ZX, ignore_index=True)
# df_SL.to_csv('{}/LN_4G_ZX.csv'.format(i),index=False)
# df_YA = pd.concat(chunks_5G_ZX, ignore_index=True)
# df_YA.to_csv('{}/LN_5G_ZX.csv'.format(i),index=False)
# df_BG = pd.concat(chunks_5G_HW, ignore_index=True)
# df_BG.to_csv('{}/LN_5G_HW.csv'.format(i),index=False)
# df_JU = pd.concat(chunks_4G_DT, ignore_index=True)
# df_JU.to_csv('{}/LN_4G_DT.csv'.format(i),index=False)
# df_WB = pd.concat(chunks_4G_NJY, ignore_index=True)
# df_WB.to_csv('{}/LN_4G_NJY.csv'.format(i),index=False)
# df_CY = pd.concat(chunks_5G_ALX, ignore_index=True)
# df_CY.to_csv('{}/LN_5G_ALX.csv'.format(i),index=False)
# print("{}分块处理之后读取用时为：%0.2f s".format(i) % (time.time() - start))

# ------------------ 生成告警日志文件 ----------------
# Data_org11 = pd.read_csv('F:\AIOps_LN\智能基站运维数据-LN/LN_5G_HW.csv',iterator=True,encoding='utf-8')
# loop = True
# chunkSize = 500000
# chunks_4G_HW = []
# num = 0
# while loop:
#     print('已执行{}行数据'.format(num))
#     try:
#         Data_org = Data_org11.get_chunk(chunkSize)
#
#         alert_dropna = Data_org.dropna(subset=['网元名称', '基站ID'])
#         alert_dropna = alert_dropna#[['一级网络类型', '网络类型', '设备类型', '基站名称', 'ENODEB_ID', '所属机房', '设备厂家名称', '设备名称', '地市名称', '告警对象名称', '告警对象设备类型','网管告警级别', '告警标题', '告警发生时间', '告警清除状态', '告警清除时间', '设备工程状态', '告警工程状态']]
#         # alert_dropna.rename(columns={'基站名称' :'网元名称'}, inplace=True)
#
#         Data_org = alert_dropna[alert_dropna['网络类型'] == '5G']
#
#         try:
#             Data_org['告警持续时间1'] = Data_org['告警发生时间'].map(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
#             Data_org['告警清除时间'] = Data_org['告警清除时间'].fillna(-1)
#             Data_org['告警持续时间2'] = Data_org['告警清除时间'].map(
#                 lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))) if x != -1 else -1)
#             Data_org['告警持续时间'] = Data_org['告警持续时间2'] - Data_org['告警持续时间1']
#             Data_org['告警持续时间'] = Data_org['告警持续时间'].map(lambda x: x if x >= 0 else 9999)
#             Data_org = Data_org.drop(['告警持续时间1', '告警持续时间2'], axis=1)
#         except:
#             Data_org['告警持续时间'] = 9999
#         # Data_org['告警日期'] = Data_org['告警发生时间'].map(lambda x: x.split()[0])
#         # Data_org['告警日期1'] = Data_org['告警日期'].map(
#         #     lambda x: datetime(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))
#
#         Data_target = Data_org#[(Data_org['告警日期1'] >= file_date) & (Data_org['告警日期1'] <= file_date)]
#         Alert_XJ = Data_target[['告警标题', '告警发生时间', '网元名称', '基站ID', '设备厂家名称', '地市名称', '告警持续时间','网络类型']]
#         Alert_XJ.columns = ['告警名称', '告警开始时间', '网元名称', '基站id', '厂家', '区域（地市）', '告警持续时间','网络类型']
#         Alert_XJ['区域（地市）'] = Alert_XJ['区域（地市）'].map(lambda x: x.split('地区')[0])
#
#         # 告警标题里 '\/' 转为 '/'
#         Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x :x.replace('\/' ,'/'))
#         try \
#             :Alert_XJ['告警名称'] = Alert_XJ['告警名称'].map(lambda x :x.replace('\\/' ,'/'))
#         except \
#             :print(11)
#
#         chunks_4G_HW.append(Alert_XJ)
#         num = num + 500000
#     except StopIteration:
#         loop = False
#         print("Iteration is stopped.")
#
# df_WN = pd.concat(chunks_4G_HW, ignore_index=True)
# df_WN = df_WN.sort_values(by='告警开始时间')
# df_WN.to_csv(r'F:\AIOps_LN\智能基站运维数据-LN/告警日志_LN_5G_HW.csv', index=None, encoding='gbk')

# ------------------- 按天分割告警日志文件 ----------------------
# # distr_list = ['西安', '咸阳', '延安', '榆林', '汉中', '宝鸡', '铜川', '安康', '商洛', '渭南']
# # for distr in distr_list:
Data_org = pd.read_csv('F:\AIOps_LN\智能基站运维数据-LN/告警日志_LN_4G_HW.csv', encoding='gbk')
Data_org['告警日期'] = Data_org['告警开始时间'].map(lambda x:str(x).split(' ')[0].replace('-',''))
date_low = '20210217'
date_high = '20210724'
date1 = pd.date_range(date_low, date_high)
date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()
for date_time in date_list:
    # Data_org[Data_org['区域（地市）'] == distr]
    date_select = Data_org[Data_org['告警日期']==date_time]
    date_select.drop(['告警日期'],inplace=True,axis=1)
    date_select['网络类型'] = 103
    date_select.to_csv('E:\PycharmProjects\LZ_Project\Data\全省\Alert_Data/告警日志{}.csv'.format(date_time),index=False,encoding='gbk')
    # date_select.to_csv('E:\PycharmProjects\LZ_Project\Data\{}\Alert_Data/告警日志{}.csv'.format(distr,date_time),index=False,encoding='gbk')
#
#
# # ------------------- 合并4G/5G文件 ----------------------
# Data_4G = pd.read_csv('F:\AIOps_LN\智能基站运维数据-LN/告警日志_LN_4G_HW.csv', encoding='gbk')
# Data_4G['网络类型'] = 103
# Data_5G = pd.read_csv('F:\AIOps_LN\智能基站运维数据-LN/告警日志_LN_5G_HW.csv', encoding='gbk')
# Data_5G['网络类型'] = 108
# Data_org = pd.concat([Data_4G,Data_5G],axis=0)
# Data_org['告警日期'] = Data_org['告警开始时间'].map(lambda x:str(x).split(' ')[0].replace('-',''))
# date_low = '20210601'
# date_high = '20210731'
# date1 = pd.date_range(date_low, date_high)
# date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()
# for date_time in date_list:
#     # Data_org[Data_org['区域（地市）'] == distr]
#     date_select = Data_org[Data_org['告警日期']==date_time]
#     date_select.drop(['告警日期'],inplace=True,axis=1)
#     date_select.to_csv('E:\PycharmProjects\LZ_Project\Data\全省\Alert_Data/告警日志{}.csv'.format(date_time),index=False,encoding='gbk')
#     # date_select.to_csv('E:\PycharmProjects\LZ_Project\Data\{}\Alert_Data/告警日志{}.csv'.format(distr,date_time),index=False,encoding='gbk')
