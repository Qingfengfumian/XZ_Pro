import pandas as pd
import datetime

distr = '汉中'
distr_dict_num = {'西安': '10', '铜川': 'TC', '宝鸡': 'BJ', '咸阳': 'XY', '渭南': '14', '汉中': 'HZ', '安康': '16',
                  '商洛': '17', '延安': '18', '榆林': '19'}
distr_ds = distr_dict_num.get(distr)
tf_jz_list = []
yc_jz_list = []
yc_prec = []
for i in range(0,60):
    predict_time_0 = datetime.datetime.strptime("2021-03-04","%Y-%m-%d")+datetime.timedelta(days=i)  # 4-27
    predict_time_1 = predict_time_0 - datetime.timedelta(days=6)
    predict_time_high = str(predict_time_0).split(' ')[0].replace('-', '')
    predict_time_low = str(predict_time_1).split(' ')[0].replace('-', '')
    predict_time = predict_time_low + '-' + predict_time_high

    TFdate = predict_time

    date_low_Time = predict_time_0+datetime.timedelta(days=0)
    TFdate = str(date_low_Time).split(' ')[0].replace('-','')
    date_high_Time = predict_time_0+datetime.timedelta(days=2)
    date = pd.date_range(str(date_low_Time).split(' ')[0].replace('-', ''),str(date_high_Time).split(' ')[0].replace('-', ''))
    date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()
    print(TFdate)
    try:
        TF_list = pd.read_csv('E:\AItask\AIOps_ShanX\服务器数据\退服\{}\OutService_{}_{}.csv'.format(distr,TFdate,distr_ds),encoding='gbk')
    except:
        continue
    # TF_list1 = TF_list[TF_list['pred_probability']>=0.6]
    TF_list1 = TF_list

    tempall = []
    for datetr in date_list:
        a = 'E:\AItask\AIOps_ShanX\服务器数据\退服\{}\告警日志{}.csv'.format(distr_ds,distr)
        Alert_date = pd.read_csv('E:\AItask\AIOps_ShanX\服务器数据\退服\{}\告警日志{}.csv'.format(distr,datetr),encoding='gbk')
        tempall.append(Alert_date)
    Alert_all = pd.concat(tempall,axis=0)
    Alert_select = Alert_all[(Alert_all['告警名称'] == '射频单元维护链路异常告警')
                                        | (Alert_all['告警名称'] == '小区不可用告警')
                                        | (Alert_all['告警名称'] == '网元连接中断')| (Alert_all['告警名称'] == 'eNodeB退服告警')| (Alert_all['告警名称'] == '传输光接口异常告警')]
    TF_count = Alert_select.groupby('基站id')['告警名称'].count()
    TF_sum = len(TF_count)
    TF_count_D = TF_count.to_frame()
    TF_count_D.reset_index(inplace=True)

    TF_merge = pd.merge(TF_list1,TF_count,on='基站id',how='left')

    # 读当天的退服
    Alert_date_now = pd.read_csv('E:\AItask\AIOps_ShanX\服务器数据\退服\{}\告警日志{}.csv'.format(distr, date_list[0]), encoding='gbk')
    Alert_select_now = Alert_date_now[(Alert_date_now['告警名称'] == '射频单元维护链路异常告警')| (Alert_date_now['告警名称'] == '小区不可用告警')| (Alert_date_now['告警名称'] == '网元连接中断') | (Alert_date_now['告警名称'] == 'eNodeB退服告警') | (Alert_date_now['告警名称'] == '传输光接口异常告警')]
    # Alert_select_now = Alert_date_now[(Alert_date_now['告警名称'] == '小区不可用告警')| (Alert_date_now['告警名称'] == '网元连接中断') | (Alert_date_now['告警名称'] == 'eNodeB退服告警')]
    TF_count_now = Alert_select_now.groupby('基站id')['告警名称'].count()
    TF_JZ_num = len(TF_count_now)

    TF_num = len(TF_merge)
    TF_merge_FN = TF_merge.fillna(-1)
    TF_1_count = TF_merge_FN['告警名称'].value_counts()
    TF_1_count_D = TF_1_count.to_frame()
    try:
        False_num = TF_1_count_D.loc[-1,'告警名称']
    except:
        False_num = 0
    if TF_num == 0:
        precision = -1
    else:
        precision = 1-(False_num/TF_num)

    TF_merge.rename(columns = {'告警名称':'退服次数'})
    TF_merge.to_csv('E:\AItask\AIOps_ShanX\服务器数据\退服\{}\Result_{}_{}_{}_{}_{}_{}.csv'.format(distr,TFdate,distr_ds,TF_sum,TF_JZ_num,TF_num,str(precision)[:6]),encoding='gbk',index=False)

    tf_jz_list.append(TF_JZ_num)
    yc_jz_list.append(TF_num)
    yc_prec.append(str(precision)[:6])

import numpy as np
aa = pd.DataFrame(np.array(tf_jz_list))
b = pd.DataFrame(np.array(yc_jz_list))
c = pd.DataFrame(np.array(yc_prec))
d = pd.concat([aa,b,c],axis=1,ignore_index=True)
d.to_csv('E:\AItask\AIOps_ShanX\服务器数据\退服\铜川\退服对比\\退服对比图表112.csv',index=False)