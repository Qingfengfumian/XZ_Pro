import pickle,time
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

from utils import mkdir, find_new_file

def dict_gen(para,distr,mode,ftype):
    start = time.clock()
    if mode=='train_XJ':
        date = para.train_date
    if mode=='predict':
        date = para.date
    AlertHW_DF2 = pd.read_csv(
                        r'./Data/{}/Alert_Deal/Samp_{}_{}/故障_处理_{}_4G.csv'.format(distr, para.mode,ftype, date),encoding='gbk',
                        engine='python')
    AlertHW_DF22 = AlertHW_DF2.reset_index(drop=True)
    AlertHW_DF = AlertHW_DF22.rename(columns={'告警开始时间':'time','基站id':'基础小区号','告警名称':'alarmname'}, inplace=False)
    AlertHW_DF['time'] = AlertHW_DF['time'].map(lambda x: x.split(' ')[0])
    try:
        AlertHW_DF['enbid+time'] = AlertHW_DF['基础小区号'].astype(int).map(str) + '|' + AlertHW_DF['time']
    except:
        AlertHW_DF['enbid+time'] = AlertHW_DF['基础小区号'].map(str) + '|' + AlertHW_DF['time']
    # ————————————————————从dataframe生成字典——————————————————————#
    AlertHW_DF_renew = AlertHW_DF[['alarmname', 'enbid+time']]
    AlertHW_DF_renew['sum_alarm'] = AlertHW_DF_renew.groupby(['enbid+time', 'alarmname'])['alarmname'].transform(
        lambda x: x.count())
    list_alarmname = AlertHW_DF_renew['alarmname'].values.tolist()
    list_enbid_time = AlertHW_DF_renew['enbid+time'].values.tolist()
    list_sum_alarm = AlertHW_DF_renew['sum_alarm'].values.tolist()
    if mode == 'train_XJ':
        Alert_Title = list(set(list_alarmname))  # 获取故障告警所有标题
        mkdir(r'./Data/{}/Inter_Result/4G/{}_告警_标题/'.format(distr, ftype))
        with open(r'./Data/{}/Inter_Result/4G/{}_告警_标题/标题_{}.txt'.format(distr, ftype, date), 'wb') as f:
            pickle.dump(Alert_Title, f)
            f.close()
        print('完成-保存{}_告警标题'.format(ftype))
    else:
        # 读取告警标题
        title_filepath = find_new_file('./Data/{}/Inter_Result/4G/{}_告警_标题/'.format(distr, ftype))
        with open(title_filepath, 'rb') as f:
            Alert_Title = pickle.load(f)
            f.close()
    Alert_HW_dict = {}
    for i in range(len(list_alarmname)):
        Alert_HW_dict.setdefault(list_enbid_time[i], {})[list_alarmname[i]] = list_sum_alarm[
            i]  # 如果键不存在于字典中，将会添加键并将值设为默认值。
        # ————————————————————从字典生成样本-故障——————————————————————#
    ColumnSet = ['enbid+date'] + Alert_Title
    Sample_HW = np.array(ColumnSet)
    len_Sample = len(Sample_HW)  # 只有title
    key_num = 0
    len_dict = len(Alert_HW_dict)
    HW_dict = []
    for key, values in Alert_HW_dict.items():
        temp_A = key_num / len_dict * 100
        HW_lst = [0 for _ in range(len_Sample)]
        HW_lst[0] = key
        for key1, values1 in values.items():
            try:
                ls_idx = ColumnSet.index(key1)  # exist title
            except:
                # print('出现新告警标题')
                continue
            HW_lst[ls_idx] = values1
        HW_dict.append(HW_lst)
        key_num += 1
    # Len_HW_dict1 = int(len(HW_dict)/4)
    # Len_HW_dict2 = int(len(HW_dict)/2)
    # Len_HW_dict3 = int(len(HW_dict)/4*3)

    # HW_np = np.array(HW_dict[:Len_HW_dict1])
    HW_np = np.array(HW_dict)
    Sample_HW_df = pd.DataFrame(HW_np, columns=ColumnSet)
    if os.path.exists('./Data/{}/Alert_Samp/Samp_{}_{}/'.format(distr,mode, ftype))==False:
        mkdir('./Data/{}/Alert_Samp/Samp_{}_{}/'.format(distr, mode,ftype))
    if mode == 'train_XJ':
        Sample_HW_df.to_csv(
        r'./Data/{}/Alert_Samp/Samp_{}_{}/故障_样本_{}_4G.csv'.format(distr, mode,ftype, date), index=False, encoding='gbk')
        print('巡检训练：{}_{}_{}样本生成 in {} seconds'.format(distr, ftype, date, (time.clock() - start)))
    if mode == 'predict':
        Sample_HW_df.to_csv(
        r'./Data/{}/Alert_Samp/Samp_{}_{}/故障_样本_{}_4G.csv'.format(distr,mode, ftype, para.date), index=False, encoding='gbk')
        print('巡检预测：{}_{}_{}样本生成 in {} seconds'.format(distr, ftype, date, (time.clock() - start)))
    # del HW_np
    # del Sample_HW_df
    # HW_np = np.array(HW_dict[Len_HW_dict1:Len_HW_dict2])
    # Sample_HW_df = pd.DataFrame(HW_np, columns=ColumnSet)
    # Sample_HW_df.to_csv(
    #         r'./Data/{}/Alert_Samp/Samp_{}_{}/故障_样本_{}1.csv'.format(distr, mode, ftype, date), index=False,
    #         encoding='gbk')
    # print('巡检训练：{}_{}_{}样本生成 in {} seconds'.format(distr, ftype, date, (time.clock() - start)))
    # del HW_np
    # del Sample_HW_df
    # HW_np = np.array(HW_dict[Len_HW_dict2:Len_HW_dict3])
    # Sample_HW_df = pd.DataFrame(HW_np, columns=ColumnSet)
    # Sample_HW_df.to_csv(
    #         r'./Data/{}/Alert_Samp/Samp_{}_{}/故障_样本_{}2.csv'.format(distr, mode, ftype, date), index=False,
    #         encoding='gbk')
    # print('巡检训练：{}_{}_{}样本生成 in {} seconds'.format(distr, ftype, date, (time.clock() - start)))
    # del HW_np
    # del Sample_HW_df
    # HW_np = np.array(HW_dict[Len_HW_dict3:])
    # Sample_HW_df = pd.DataFrame(HW_np, columns=ColumnSet)
    # Sample_HW_df.to_csv(
    #         r'./Data/{}/Alert_Samp/Samp_{}_{}/故障_样本_{}3.csv'.format(distr, mode, ftype, date), index=False,
    #         encoding='gbk')
    # print('巡检训练：{}_{}_{}样本生成 in {} seconds'.format(distr, ftype, date, (time.clock() - start)))
    # del HW_np
    # del Sample_HW_df

def proc_XJ_4G(para):
    distr_list = para.distr_list
    mode = para.mode
    ftype = para.ftype
    for distr in distr_list:
        dict_gen(para, distr, mode, ftype)

