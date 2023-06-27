import pandas as pd
import datetime
# nrm_path = '/data/resources/nrm/'
# disti_path = '/home/AIOps/LN_Project/Data/汇总/Project_Opt/'

nrm_path = 'E:\AI项目组材料\项目材料/202001网络智能化战略项目/202207西藏智能运维\数据调研/20230113资源数据'
disti_path = nrm_path

now_date = datetime.datetime.now()-datetime.timedelta(days=3)
# now_date_str = str(now_date).split(' ')[0].replace('-','')

now_date_str = '20230201'

nrm_4G = pd.read_csv(nrm_path+'/4G_enodeb_{}.csv'.format(now_date_str),sep=',')
print(nrm_path+'/4G_enodeb_{}.csv'.format(now_date_str))
nrm_5G = pd.read_csv(nrm_path+'/5G_gnodeb_{}.csv'.format(now_date_str),sep=',')
print(nrm_path+'/5G_gnodeb_{}.csv'.format(now_date_str))

nrm_4G_sel = nrm_4G[['NODEB_ID','USERLABEL','RELATED_ROOM_LOCATION','LIFECYCLE_STATUS','DEVICE_TYPE','BEEHIVE_TYPE','VIP_TYPE','VENDOR_ID','CITY_ID','COUNTY_ID','LONGITUDE','LATITUDE','RMUID']]
nrm_4G_sel['网络制式'] = '4G'
nrm_4G_sel['NODEB_ID'] = nrm_4G_sel['NODEB_ID'].map(lambda x:x.replace('XZ-',''))
nrm_5G_sel = nrm_5G[['NODEB_ID','USERLABEL','RELATED_ROOM_LOCATION','LIFECYCLE_STATUS','DEVICE_TYPE','BEEHIVE_TYPE','VIP_TYPE','VENDOR_ID','CITY_ID','COUNTY_ID','LONGITUDE','LATITUDE','RMUID']]
nrm_5G_sel['网络制式'] = '5G'

nrm_sum = pd.concat([nrm_4G_sel,nrm_5G_sel])

nrm_sum.columns = ['ENODEB_ID','基站名称','所属机房/位置点','生命周期状态','设备类型','覆盖类型','VIP级别','设备厂家','所属地市','所属区县','经度','纬度','RMUID','所属业务系统']
nrm_sum = nrm_sum[['基站名称','VIP级别','覆盖类型','设备厂家','生命周期状态','所属地市','所属机房/位置点','所属区县','经度','纬度','ENODEB_ID','设备类型','RMUID','所属业务系统']]

# nrm_sum_select.columns = ['基站名称','所属机房/位置点','覆盖类型','VIP级别','所属地市','所属区县','设备厂家','生命周期状态','ENODEB_ID','小区中文名','所属业务系统']
nrm_sum.dropna(subset=['ENODEB_ID'],axis=0,inplace=True)

nrm_sum.to_csv(disti_path+'/GC-CELL-BS-ROOM.csv',index=False,encoding='gbk')

# path = 'E:\PycharmProjects/NM_Pro/site_20220624.csv'
#
# a = pd.read_csv(path,sep='|')
# a.to_csv(path+'new',index=False)