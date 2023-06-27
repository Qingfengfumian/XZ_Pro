import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
from collections import Counter
import datetime
import time
import pickle

select_cols = ['告警标准名', '发生时间', '网元', '设备厂家', '归属地市', '告警发现时间', '清除时间', '机房名称']
rename_cols = ['告警名称', '告警开始时间', '网元名称', '厂家', '区域（地市）', '告警发现时间', '清除时间', '机房名称']

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\") # 删除 string 字符串末尾的指定字符
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def find_new_file(dir):
    '''查找目录下最新的文件'''
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    file = os.path.join(dir, file_lists[-1])
    return file

def read_excel_to_csv(basedir, to_dir, filename, outputname):
    print(time.ctime())
    xl = pd.ExcelFile(os.path.join(basedir, filename))
    print(time.ctime())
    df_list = []
    for sheet_name in xl.sheet_names:
        df_tmp = xl.parse(sheet_name)
        df_tmp = df_tmp[select_cols]
        df_list.append(df_tmp)
        print(time.ctime())
    df_all_sheet = pd.concat(df_list)
    df_all_sheet.columns = rename_cols
    df_all_sheet.to_csv(os.path.join(to_dir, outputname), index=False)
    return df_all_sheet


def dict_Gen_part(AlertHW_DF2, orgpath, dist, date, ftype, alertclass, is_train=True, IFpre=False,sam_dir='AIOps_all'):
    # AIOps/Alert_Deal/城一/Deal_20190923-1008/城一_中兴_故障_处理_20190923-1008.csv
    start = time.clock()
    # ————————————————————生成故障字典——————————————————————#
    AlertHW = []
    ColumnLable = ['time', 'alarmname', 'enbname', '基础小区号', '厂家']
    # AlertHW_DF2 = pd.read_csv(r'D:/working_document/AIOps/Alert_Deal/怀柔/Deal_20191008-1014/怀柔_中兴_故障_处理_20191008-1014.csv'.format(dist,date,dist,ftype,alertclass,date),encoding='utf-8',engine='python')
    # AlertHW_DF3 = pd.concat([AlertHW_DF1,AlertHW_DF2],axis=0)
    if AlertHW_DF2.shape[0] == 0:
        print('{}_{}_{}_样本为空'.format(dist, ftype, alertclass))
    else:
        AlertHW_DF = AlertHW_DF2.reset_index(drop=True)
        # 按天统计  将time截取剩day
        AlertHW_DF_Pre = AlertHW_DF.copy()
        AlertHW_DF['time'] = AlertHW_DF['time'].map(lambda x: x.split(' ')[0])

        # 以enbid和time为key
        AlertHW_DF['enbid+time'] = AlertHW_DF['基础小区号'].astype(str).map(str) + '|' + AlertHW_DF['time']

        # ————————————————————从dataframe生成字典——————————————————————#
        AlertHW_DF_renew = AlertHW_DF[['alarmname', 'enbid+time']]
        AlertHW_DF_renew['sum_alarm'] = AlertHW_DF_renew.groupby(['enbid+time', 'alarmname'])['alarmname'].transform(
            lambda x: x.count())

        # print('HW_GZ字典dataframe生成 in %s seconds' % (time.clock() - start))

        list_alarmname = AlertHW_DF_renew['alarmname'].values.tolist()
        list_enbid_time = AlertHW_DF_renew['enbid+time'].values.tolist()
        list_sum_alarm = AlertHW_DF_renew['sum_alarm'].values.tolist()
        # Alert_HW_dict = dict(zip(list_enbid_time,zip(list_alarmname,list_sum_alarm)))
        if ((is_train is True) and (dist == '汇总')):
            # 保存告警标题
            Alert_Title = list(set(list_alarmname))  # 获取故障告警所有标题
            mkdir('{}{}/Inter_Result/{}_{}_标题/'.format(orgpath,sam_dir, ftype, alertclass))
            with open('{}{}/Inter_Result/{}_{}_标题/{}_{}_标题_{}.txt'.format(orgpath,sam_dir, ftype, alertclass, ftype,
                                                                                  alertclass, date), 'wb') as f:
                pickle.dump(Alert_Title, f)
                f.close()
            print('完成-保存{}_{}_告警标题'.format(ftype, alertclass))
            with open(
                    '{}{}/Inter_Result/{}_{}_标题/{}_{}_标题_{}.txt'.format(orgpath,sam_dir, ftype, alertclass, ftype,
                                                                                     alertclass, date), 'wb') as f1:
                pickle.dump(Alert_Title, f1)
                f1.close()
            print('完成-转存{}_{}_告警标题'.format(ftype, alertclass))
        else:
            # 读取告警标题
            title_filepath = find_new_file('{}{}/Inter_Result/{}_{}_标题/'.format(orgpath,sam_dir, ftype, alertclass))
            with open(title_filepath, 'rb') as f:
                Alert_Title = pickle.load(f)
                f.close()
        Alert_HW_dict = {}
        for i in range(len(list_alarmname)):
            Alert_HW_dict.setdefault(list_enbid_time[i], {})[list_alarmname[i]] = list_sum_alarm[
                i]  # 如果键不存在于字典中，将会添加键并将值设为默认值。
        # print('字典dict生成 in %s seconds' % (time.clock() - start))
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

        HW_np = np.array(HW_dict)
        Sample_HW_df = pd.DataFrame(HW_np, columns=ColumnSet)
        mkdir('{}{}/Alert_Samp/{}/Samp_{}/'.format(orgpath,sam_dir, dist, date))
        Sample_HW_df.to_csv(
            '{}{}/Alert_Samp/{}/Samp_{}/{}_{}_{}_样本_{}.csv'.format(orgpath,sam_dir, dist, date, dist, ftype, alertclass,
                                                                           date), index=False, encoding='gbk')
        print('{}_{}_{}_样本生成 in {} seconds'.format(dist, ftype, alertclass, (time.clock() - start)))
        if IFpre == True:
            if dist == '汇总':
                starttime = time.clock()
                # ————————————————————生成预测样本——————————————————————#
                AlertHW_DF_Pre['time'] = AlertHW_DF_Pre['time'].map(lambda x: x.split(':')[0])  # 按小时统计
                # 以enbid和time为key
                AlertHW_DF_Pre['enbid+time'] = AlertHW_DF_Pre['基础小区号'].astype(int).map(str) + '|' + AlertHW_DF_Pre[
                    'time']
                # ————————————————————从dataframe生成字典——————————————————————#
                AlertHW_DF_renew = AlertHW_DF_Pre[['alarmname', 'enbid+time']]
                AlertHW_DF_renew['sum_alarm'] = AlertHW_DF_renew.groupby(['enbid+time', 'alarmname'])[
                    'alarmname'].transform(lambda x: x.count())
                list_alarmname = AlertHW_DF_renew['alarmname'].values.tolist()
                list_enbid_time = AlertHW_DF_renew['enbid+time'].values.tolist()
                list_sum_alarm = AlertHW_DF_renew['sum_alarm'].values.tolist()
                # Alert_HW_dict = dict(zip(list_enbid_time,zip(list_alarmname,list_sum_alarm)))
                title_filepath = find_new_file('{}{}/Inter_Result/{}_{}_标题/'.format(orgpath,sam_dir, ftype, alertclass))
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

                HW_np = np.array(HW_dict)
                Sample_HW_df = pd.DataFrame(HW_np, columns=ColumnSet)
                # ————————————————————取最后三天的数据——————————————————————#
                last_date = date[:4] + '-' + date[-4:-2] + '-' + date[-2:]
                timeArray = datetime.datetime.strptime(last_date, "%Y-%m-%d")
                delta1 = datetime.timedelta(days=1)
                delta2 = datetime.timedelta(days=2)
                last_date1 = (timeArray - delta1).strftime('%Y-%m-%d')
                last_date2 = (timeArray - delta2).strftime('%Y-%m-%d')
                search = '|'.join([last_date, last_date1, last_date2])
                bool = Sample_HW_df['enbid+date'].str.contains(search)
                Sample_HW_Select = Sample_HW_df[bool]
                Sample_HW_sort = Sample_HW_Select.sort_values(by='enbid+date', ascending=True)
                Sample_HW_sort['enbid'] = Sample_HW_sort['enbid+date'].map(lambda x: x.split('|')[0])
                Sample_HW_sort['date'] = Sample_HW_sort['enbid+date'].map(lambda x: x.split('|')[1])
                # 将告警按基站分为不同表格，并按csv格式输出
                # 计数每个基站出现的告警数，由于告警表已排序，因此可按告警出现次数大块输出
                CT = Counter(Sample_HW_sort['enbid'])
                CT_sort = sorted(CT.items(), key=lambda d: d[0])
                Sample_HW_sort = Sample_HW_sort.drop(['enbid', 'date'], axis=1)
                # ————————————————————告警表分基站输出——————————————————————#
                mkdir('{}{}/Alert_Samp/{}/Samp_{}/Pre_{}_{}/'.format(orgpath,sam_dir, dist, date, ftype, alertclass))
                start = 0
                index = 1
                for key in CT_sort:
                    end = key[1] + start
                    AlertHW_DF = Sample_HW_sort[start:end]
                    # 输出成csv表格
                    AlertHW_DF_idx = AlertHW_DF['enbid+date'].values[0].split('|')[0] + '|' + last_date
                    AlertHW_DF_idx1 = AlertHW_DF['enbid+date'].values[0].split('|')[0] + '|' + last_date1
                    AlertHW_DF_idx2 = AlertHW_DF['enbid+date'].values[0].split('|')[0] + '|' + last_date2

                    new_table = np.zeros([72, len(Alert_Title) + 1], dtype=int)
                    table_Dataframe = pd.DataFrame(new_table, columns=AlertHW_DF.columns)
                    # 新建24行全0的dataframe
                    range_ser = (['0' + str(x) for x in range(10)] + [str(x) for x in range(10, 24)]) * 3
                    table_Dataframe['enbid+date'] = range_ser
                    table_Dataframe['enbid+date'][0:24] = table_Dataframe['enbid+date'][0:24].map(
                        lambda x: AlertHW_DF_idx2 + ' ' + x)
                    table_Dataframe['enbid+date'][24:48] = table_Dataframe['enbid+date'][24:48].map(
                        lambda x: AlertHW_DF_idx1 + ' ' + x)
                    table_Dataframe['enbid+date'][48:72] = table_Dataframe['enbid+date'][48:72].map(
                        lambda x: AlertHW_DF_idx + ' ' + x)
                    AlertHW_DF_contact = pd.concat([AlertHW_DF, table_Dataframe], axis=0)
                    AlertHW_DF_delet = AlertHW_DF_contact.drop_duplicates(subset=['enbid+date'], keep='first',
                                                                          inplace=False)
                    AlertHW_DF_delet_sort = AlertHW_DF_delet.sort_values(by='enbid+date', ascending=True)
                    AlertHW_DF_delet_sort.to_csv(
                        '{}{}/Alert_Samp/{}/Samp_{}/Pre_{}_{}/AlertHW_DF_{}.csv'.format(orgpath,sam_dir, dist, date,
                                                                                                ftype,
                                                                                                alertclass, key[0]),
                        index=None)

                    index += 1
                    start = end
                print('{}预测样本生成 in {}s'.format(ftype, time.clock() - starttime))


def excel_Alert_data_convert(wuxian_base_dir,orgpath ,start, end, year,gc='4G'):
    time_list = list(pd.date_range(pd.to_datetime(year + start), pd.to_datetime(year + end), freq='D').strftime('%m%d'))
    date = year + start + '-' + end
    df_list = []
    start_date = datetime.datetime(int(year), int(start[:2]), int(start[-2:]))
    #start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
    delta = datetime.timedelta(days=7)
    end_date = start_date + delta


    df_concat = pd.read_csv(open(os.path.join(wuxian_base_dir,'ZJ_wuxian_alarm_'+gc+'.csv'),encoding='utf-8'))
    df_concat_1 = df_concat.copy()
    df_concat['告警开始时间'] = pd.to_datetime(df_concat['告警开始时间'])
    df_concat = df_concat[(df_concat['告警开始时间'] >= start_date) & (df_concat['告警开始时间'] < end_date)]
    df_concat['告警开始时间'] = df_concat['告警开始时间'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    print('读取数据成功!!', len(df_concat))
    if len(df_concat)==0:
        df_concat = df_concat_1
        df_concat['告警开始时间'] = df_concat['告警开始时间'].apply(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d %H:%M'))
        df_concat = df_concat[(df_concat['告警开始时间']>=start_date)&(df_concat['告警开始时间']<end_date)]
        try:
            df_concat['告警开始时间'] = df_concat['告警开始时间'].apply(lambda x: datetime.datetime.strftime(x, '%Y/%m/%d %H:%M:%S'))
        except:
            df_concat['告警开始时间'] = df_concat['告警开始时间'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))

    df_concat = df_concat.dropna(axis=0, thresh=None, subset=['告警名称'], inplace=False)
    df_concat['告警开始小时'] = df_concat['告警开始时间'].map(lambda x: x.split(' ')[1].split(':')[0])
    df_concat = df_concat[~df_concat['告警开始小时'].isin(['23', '00', '01', '02', '03', '04', '05', '06'])]
    df_concat = df_concat.drop(['告警开始小时'], axis=1)
    df_concat['网元名称'].fillna('A网元', inplace=True)
    df_concat['清除时间'].fillna('F', inplace=True)
    # df_concat['机房名称'].fillna('A机房', inplace=True)
    df_concat['区域（地市）'].fillna('A地市', inplace=True)
    df1_wangyuan = df_concat[['网元名称']]
    df1_wangyuan = df1_wangyuan.drop_duplicates(['网元名称'])
    list2 = [10000000 + i for i in range(len(df1_wangyuan['网元名称']))]
    df1_wangyuan['基站id'] = list2
    df1_wangyuan['基站id'] = df1_wangyuan['基站id'].astype('str')
    df1_merge = pd.merge(df_concat, df1_wangyuan, how='left')
    df1_merge['基站id'] = df1_merge['基站id'].astype('str')
    df1_merge_noclear = df1_merge[df1_merge['清除时间'].isin(['F'])]

    df1_merge = df1_merge[['告警开始时间', '告警名称', '基站id', '网元名称', '厂家', '区域（地市）']]
    df1_merge = df1_merge.dropna(subset=['网元名称'])
    df1_merge = df1_merge.reset_index(drop=True)
    df1_merge.columns = ['time', 'alarmname', '基础小区号', 'enbname', '厂家', '区域（地市）']
    # df1_merge = pd.merge(df1_merge, alarm_data, how='left')
    # print(df1_merge['网管告警级别'].value_counts())
    df1_merge_noclear = df1_merge_noclear[['告警开始时间', '告警名称', '基站id', '网元名称', '厂家', '区域（地市）']]
    df1_merge_noclear = df1_merge_noclear.dropna(subset=['网元名称'])
    df1_merge_noclear = df1_merge_noclear.reset_index(drop=True)
    df1_merge_noclear.columns = ['time', 'alarmname', '基础小区号', 'enbname', '厂家', '区域（地市）']

    Dists = ['合肥','芜湖','蚌埠','淮南','马鞍山','淮北','铜陵','安庆','黄山','滁州','阜阳','宿州','六安','亳州','池州','宣城']
    ftypes = ['华为', '中兴', '诺基亚', '爱立信']
    #ftypes = ['华为', '中兴', '诺基亚']

    for df_ftype in df1_merge.groupby(['厂家']):
        if df_ftype[0] in ftypes:
            tmp_df_ftype = df_ftype[1]
            dict_Gen_part(tmp_df_ftype, orgpath, '汇总', date, df_ftype[0], alertclass='故障', is_train=True, IFpre=False,sam_dir='AIOps_all_'+gc)
            for tmp_df_dist in tmp_df_ftype.groupby(['区域（地市）']):
                if tmp_df_dist[0] in Dists:
                    dict_Gen_part(tmp_df_dist[1], orgpath, tmp_df_dist[0], date, df_ftype[0], alertclass='故障',
                                  is_train=True, IFpre=False,sam_dir='AIOps_all_'+gc)
    for df_ftype in df1_merge_noclear.groupby(['厂家']):
        if df_ftype[0] in ftypes:
            tmp_df_ftype = df_ftype[1]
            dict_Gen_part(tmp_df_ftype, orgpath, '汇总', date, df_ftype[0], alertclass='故障', is_train=True, IFpre=False,sam_dir='AIOps_no_clear_'+gc)
            for tmp_df_dist in tmp_df_ftype.groupby(['区域（地市）']):
                if tmp_df_dist[0] in Dists:
                    dict_Gen_part(tmp_df_dist[1], orgpath, tmp_df_dist[0], date, df_ftype[0], alertclass='故障',
                                  is_train=True, IFpre=False,sam_dir='AIOps_no_clear_'+gc)


if __name__ == '__main__':
    # 传参进：文件目录、文件输出目录、文件名、输出文件名
    wuxian_base_dir = '/home/share/ch/移动设计院/data/sample/告警统计/A全量告警1014-1018/无线/'
    wug_base_dir = '/home/share/ch/移动设计院/data/sample/告警统计/A全量告警1014-1018/5G/'
    start = '1014'
    end = '1018'
    year = '2020'
    orgpath = r"/home/share/ch/移动设计院/data/sample/告警统计/tmp/"
    excel_Alert_data_convert(wuxian_base_dir=wuxian_base_dir,orgpath= orgpath,start=start, end=end, year=year,gc='4G')
    excel_Alert_data_convert(wuxian_base_dir=wug_base_dir,orgpath =orgpath,start=start, end=end, year=year,gc='5G')
