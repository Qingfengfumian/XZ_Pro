import pandas as pd
import time


# path = 'E:\PycharmProjects\BS_Monitor_V1\Data\汇总\Alert_Data\\'
# for i in ['alarm_20200701_20200731.csv','alarm_20200901_20200930.csv']:
# ------------- 读原始告警数据 分块处理 -------------
i = 'F:\AIOps_ShanX\智能基站运维数据-LN/'
start = time.time()
yy = pd.read_csv('{}/告警表：ALARM_RAW_ALARMS.csv'.format(i), iterator=True, encoding='utf-8')
# df = yy.get_chunk(200000)

loop = True
chunkSize = 1000000
chunks_4G_HW = []
chunks_4G_ALX = []
chunks_4G_ZX = []
chunks_5G_ZX = []
chunks_5G_HW = []
chunks_4G_DT = []
chunks_4G_NJY = []
chunks_5G_ALX = []
num = 0
while loop:
    print('已执行{}行数据'.format(num))
    try:
        linshidata = yy.get_chunk(chunkSize)
        linshidata = linshidata[['告警标题','告警发生时间','告警清除时间','网元名称','基站ID','设备厂家名称','地市名称','网络类型']]
        linshidatab = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '华为')]
        chunks_4G_HW.append(linshidatab)
        linshidatac = linshidata[(linshidata['网络类型'] == '4G')  & (linshidata['设备厂家名称'] == '爱立信')]
        chunks_4G_ALX.append(linshidatac)
        linshidatad = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '中兴')]
        chunks_4G_ZX.append(linshidatad)
        linshidatae = linshidata[(linshidata['网络类型'] == '5G') & (linshidata['设备厂家名称'] == '中兴')]
        chunks_5G_ZX.append(linshidatae)
        linshidatae = linshidata[(linshidata['网络类型'] == '5G') & (linshidata['设备厂家名称'] == '华为')]
        chunks_5G_HW.append(linshidatae)
        linshidatae = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '大唐')]
        chunks_4G_DT.append(linshidatae)
        linshidatae = linshidata[(linshidata['网络类型'] == '4G') & (linshidata['设备厂家名称'] == '诺基亚')]
        chunks_4G_NJY.append(linshidatae)
        linshidatae = linshidata[(linshidata['网络类型'] == '5G') & (linshidata['设备厂家名称'] == '爱立信')]
        chunks_5G_ALX.append(linshidatae)
        num = num + 1000000
    except StopIteration:
        loop = False
        print("Iteration is stopped.")

df_WN = pd.concat(chunks_4G_HW, ignore_index=True)
df_WN.to_csv('{}/LN_4G_HW.csv'.format(i),index=False)
df_AK = pd.concat(chunks_4G_ALX, ignore_index=True)
df_AK.to_csv('{}/LN_4G_ALX.csv'.format(i),index=False)
df_SL = pd.concat(chunks_4G_ZX, ignore_index=True)
df_SL.to_csv('{}/LN_4G_ZX.csv'.format(i),index=False)
df_YA = pd.concat(chunks_5G_ZX, ignore_index=True)
df_YA.to_csv('{}/LN_5G_ZX.csv'.format(i),index=False)
df_BG = pd.concat(chunks_5G_HW, ignore_index=True)
df_BG.to_csv('{}/LN_5G_HW.csv'.format(i),index=False)
df_JU = pd.concat(chunks_4G_DT, ignore_index=True)
df_JU.to_csv('{}/LN_4G_DT.csv'.format(i),index=False)
df_WB = pd.concat(chunks_4G_NJY, ignore_index=True)
df_WB.to_csv('{}/LN_4G_NJY.csv'.format(i),index=False)
df_CY = pd.concat(chunks_5G_ALX, ignore_index=True)
df_CY.to_csv('{}/LN_5G_ALX.csv'.format(i),index=False)
print("{}分块处理之后读取用时为：%0.2f s".format(i) % (time.time() - start))



# import pandas as pd
#
# path = 'E:\PycharmProjects\BS_Monitor_V1\Data\汇总\Alert_Data\\'
# for i in ['{}alarm_20210226_time.csv'.format(path)]:#,'alarm_20200901_20200930.csv','alarm_20201101_20201130.csv','alarm_20200801_20200831.csv','alarm_20201001_20201031.csv','alarm_20201201_20201231.csv']:
#     print(i)
#     try:
#         linshidata = pd.read_csv(i,encoding='utf-8',nrows = 400000)
#     except:
#         linshidata = pd.read_csv(i,encoding='gbk',nrows = 400000)
#     linshidatab = linshidata[(linshidata['网络类型'] == 103) & (linshidata['地市名称'] == '西安地区') & (linshidata['设备厂家名称'] == '华为')]
#     try:
#         linshidata = pd.read_csv(i,encoding='utf-8',skiprows = range(1,400000))
#     except:
#         linshidata = pd.read_csv(i,encoding='gbk',skiprows = range(1,400000))
#     linshidatac = linshidata[(linshidata['网络类型'] == 103) & (linshidata['地市名称'] == '西安地区') & (linshidata['设备厂家名称'] == '华为')]
#     linshidatad = pd.concat([linshidatab,linshidatac])
#     linshidatad.to_csv('{}xian_xian.csv'.format(path),index=False)
