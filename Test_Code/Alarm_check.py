import pandas as pd
import os

# alarm_path = 'E:/AI项目组材料/项目材料/202001网络智能化战略项目/202112内蒙智能运维/NM_数据/告警数据/'
# middle_path = 'E:/AI项目组材料/项目材料/202001网络智能化战略项目/202112内蒙智能运维/NM_数据/告警数据/Deal/'
# result_path = 'E:/AI项目组材料/项目材料/202001网络智能化战略项目/202112内蒙智能运维/NM_数据/告警数据/Result/'
alarm_path = '/data/alarm_hour_wuxiandonghuan/'
middle_path = '/home/data/alarm_deal/Deal/'
result_path = '/home/data/alarm_deal/Result/'

for alarm_Cir in os.listdir(alarm_path):

    # alarm_S = 'ALARM-20220514'
    print(alarm_Cir)
    try:alarm_Data = pd.read_csv(alarm_path+'/{}'.format(alarm_Cir),encoding='gbk')
    except:alarm_Data = pd.read_csv(alarm_path+'/{}'.format(alarm_Cir),encoding='gb18030')

    alarm_Sel = alarm_Data[['省份','发生时间']]

    alarm_Sel['小时'] = alarm_Data['发生时间'].map(lambda x:x.split(':')[0])

    alarm_Group = alarm_Sel.groupby(['小时'],as_index=False)['省份'].count()
    alarm_Group.rename(columns={'省份':alarm_Cir.split('.')[0]},inplace=True)
    alarm_Group['排序'] = alarm_Group['小时'].map(lambda x:int(x.split(' ')[1]))
    alarm_Group.sort_values(by='排序',inplace=True)


    alarm_Group.to_csv(middle_path+'/Deal-{}'.format(alarm_Cir),encoding='gbk',index=False)

deal_data = []
for deal_Cir in os.listdir(middle_path):

    # alarm_S = 'ALARM-20220514'
    print(deal_Cir)
    alarm_Data = pd.read_csv(middle_path+'/{}'.format(deal_Cir),encoding='gbk',header=None)
    deal_data.append(alarm_Data)

deal_Con = pd.concat(deal_data,axis=0)
deal_Con.to_csv(result_path+'/Final_deal.csv',encoding='gbk',index=False)
