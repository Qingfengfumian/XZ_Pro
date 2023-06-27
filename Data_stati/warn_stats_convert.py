import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
import datetime
import time
from tools import param_con
from excel_read_convert import excel_Alert_data_convert
import warnings
import re

warnings.filterwarnings("ignore")


def read_node_num(num_path):
    xl = pd.read_excel(num_path)
    xl.columns = ['地市', '5G_基站数', '5G_载频数', '5G_小区数', 'TD-LTE_基站数', 'TD-LTE_载频数', 'TD-LTE_小区数', 'FDD_基站数', 'FDD_载频数',
                  'FDD_小区数']
    xl.drop(0, inplace=True)
    xl['4G_基站数'] = xl['TD-LTE_基站数'] + xl['FDD_基站数']
    dict_5G = xl.set_index('地市')['5G_基站数'].to_dict()
    dict_4G = xl.set_index('地市')['4G_基站数'].to_dict()
    return dict_4G, dict_5G


def get_warn_class_map(warn_table_dir, group):
    if group == '4G':
        df_alarm_class = pd.read_excel(warn_table_dir, sheet_name='4G')
    else:
        df_alarm_class = pd.read_excel(warn_table_dir, sheet_name='5G')
    df_alarm_class.dropna(subset=['专题大类'], inplace=True)
    df_alarm_class = df_alarm_class[['厂家', '告警标题', '专题大类']]
    class_list = list(df_alarm_class.专题大类.unique())
    warn_class_map = dict(list(df_alarm_class.groupby(['厂家'])))
    class_map = {}
    for mr in warn_class_map.keys():
        class_map_t = {}
        for tmp_class in warn_class_map[mr].groupby('专题大类'):
            for warn in set(tmp_class[1][['告警标题']].values.reshape(-1)):
                class_map_t[warn] = tmp_class[0]
        class_map[mr] = class_map_t
    return class_map, class_list


def stats_group_convert(base_dir, to_dir, df_name, mfrs, class_map):
    tmp_class_map = class_map[mfrs]
    data_stats_tmp = pd.read_csv(os.path.join(base_dir, df_name), engine='python', encoding='GBK')
    data_stats_tmp['enbid'] = data_stats_tmp['enbid+date'].str.split('|', expand=True)[0]
    data_stats_tmp.drop(['enbid+date'], axis=1, inplace=True)
    stats_fir = data_stats_tmp.groupby('enbid').sum().reset_index()
    stats_fir.to_csv(os.path.join(to_dir, '日期合并_' + df_name), index=False, encoding='utf_8_sig')
    stats_fir_tmp = stats_fir.set_index('enbid').T
    stats_fir_tmp['warn'] = stats_fir_tmp.index
    stats_fir_tmp['warn_class'] = stats_fir_tmp.warn.apply(
        lambda x: tmp_class_map[x] if x in tmp_class_map.keys() else x)
    tmp_result = stats_fir_tmp.drop('warn', axis=1).groupby('warn_class').sum()
    try:
        result = tmp_result.loc[list(set(tmp_class_map.values()))].T.reset_index()
    except KeyError:
        columns1 = list(set(tmp_class_map.values()))
        columns1.insert(0, 'enbid')
        result = pd.DataFrame(columns=columns1)
        # result = tmp_result.loc[list(set(tmp_class_map.values()))]
    result.to_csv(os.path.join(to_dir, 'warn_class_unit_' + df_name), index=False, encoding='utf_8_sig')
    return result


def alert_stat_with_city(time_key, alert_num, warn_table_dir, alert_all_base, alert_noclear_base, alert_to_base,
                         Alert_all_convert_base, Alert_no_clear_convert_base, group='4G'):
    class_mapping, alert_class_list = get_warn_class_map(warn_table_dir, group)
    select_cols = ['地市', '告警已消除基站数', '告警未消除基站数', '总计基站数']
    rename_dict = {'告警已消除基站数': group + '_告警已消除基站数', '告警未消除基站数': group + '_告警未消除基站数', '总计基站数': group + '_总计基站数'}
    cls_dict = {}
    for cls in alert_class_list:
        cls_dict[cls] = pd.DataFrame([], columns=['地市', '厂家', '告警已消除基站数', '告警未消除基站数', '总计基站数'])
    print('遍历所有告警文件')
    for root, dirs, files in tqdm(os.walk(alert_all_base)):
        for file in files:
            if time_key in file:
                tmp_dict = {'地市': 0, '厂家': 0, '告警已消除基站数': 0, '告警未消除基站数': 0, '总计基站数': 0}
                # print(root,file)
                file_split = file.split('_')
                city = file_split[0]
                mfrs = file_split[1]
                # if group=='5G' and mfrs in ['爱立信', '诺基亚']:
                if group=='5G' and mfrs in ['诺基亚']:
                    continue
                if not os.path.exists(root.replace(alert_all_base, Alert_all_convert_base)):
                    os.makedirs(root.replace(alert_all_base, Alert_all_convert_base))
                warn_class = stats_group_convert(root, root.replace(alert_all_base, Alert_all_convert_base), file, mfrs,
                                                 class_mapping)
                num_alert = (warn_class[list(set(class_mapping[mfrs].values()))] > 0).astype(int).sum(axis=0)
                for cls in num_alert.index:
                    tmp_dict['地市'] = city
                    tmp_dict['厂家'] = mfrs
                    tmp_dict['总计基站数'] = num_alert[cls]
                    cls_dict[cls] = cls_dict[cls].append(tmp_dict, ignore_index=True)
    print('遍历未消除告警文件')
    for root, dirs, files in tqdm(os.walk(alert_noclear_base)):
        for file in files:
            if time_key in file:
                # print(root,file)
                file_split = file.split('_')
                city = file_split[0]
                mfrs = file_split[1]
                # if group=='5G' and mfrs in ['爱立信', '诺基亚']:
                if group == '5G' and mfrs in ['诺基亚']:
                    continue
                if not os.path.exists(root.replace(alert_noclear_base, Alert_no_clear_convert_base)):
                    os.makedirs(root.replace(alert_noclear_base, Alert_no_clear_convert_base))
                warn_class = stats_group_convert(root, root.replace(alert_noclear_base, Alert_no_clear_convert_base),
                                                 file, mfrs, class_mapping)
                num_alert = (warn_class[list(set(class_mapping[mfrs].values()))] > 0).astype(int).sum(axis=0)
                for cls in num_alert.index:
                    cls_dict[cls].loc[(cls_dict[cls]['地市'] == city) & (cls_dict[cls]['厂家'] == mfrs), '告警未消除基站数'] = \
                        num_alert[cls]

    concat_df = pd.DataFrame()
    concat_df_5G = pd.DataFrame()
    for cls in alert_class_list:
        if cls == 'OMC':
            print()
        cls_dict[cls]['告警已消除基站数'] = cls_dict[cls]['总计基站数'] - cls_dict[cls]['告警未消除基站数']
        tmp_re = cls_dict[cls].groupby('地市').sum()
        tmp_re.reset_index(inplace=True)
        all_city_sum = tmp_re[tmp_re['地市'] == '汇总']
        all_city_sum.loc[:, '地市'] = '全省'
        tmp_re.drop(tmp_re[tmp_re['地市'] == '汇总'].index, axis=0, inplace=True)
        tmp_re = tmp_re.append(all_city_sum, ignore_index=True)
        tmp_re = tmp_re[select_cols].rename(columns=rename_dict)
        if group == '4G':
            if not os.path.exists(os.path.join(alert_to_base, time_key)):
                os.makedirs(os.path.join(alert_to_base, time_key))
            # tmp_re.to_csv(
            #     os.path.join(alert_to_base, time_key, '基站数统计', time_key + '_' + cls + '_告警统计.csv'), index=False,
            #     encoding='utf_8_sig')
            tmp_re['告警类别'] = cls
            concat_df = pd.concat([concat_df, tmp_re])
        if group == '5G':
            tmp_re['告警类别'] = cls
            concat_df_5G = pd.concat([concat_df_5G, tmp_re])
    if group == '4G':
        analyse_start_date, analyse_end_date = get_current_week()
        concat_df['统计周期开始时间'] = str(analyse_start_date)
        concat_df['统计周期结束时间'] = str(analyse_end_date)
        concat_df = concat_df[['统计周期开始时间', '统计周期结束时间', '地市', '告警类别', '4G_告警已消除基站数', '4G_告警未消除基站数', '4G_总计基站数']]
        concat_df.to_csv(os.path.join(alert_to_base, time_key, time_key + '_bs_stati.csv'), index=False, encoding='utf_8')
        file_path = os.path.join(alert_to_base, time_key, time_key + '_bs_stati.csv')
        cmd = 'scp {} root@82.221.9.85:{}'.format(file_path, '/data/resources/result_stati/')
        print(cmd)
        os.system(cmd)
    if group == '5G':
        tmp_re_4G = pd.read_csv(os.path.join(alert_to_base, time_key, time_key + '_bs_stati.csv'), engine='python', encoding='utf_8')
        concat_df_5G = concat_df_5G[['地市', '告警类别', '5G_告警已消除基站数', '5G_告警未消除基站数', '5G_总计基站数']]
        concat_df_45G = pd.merge(tmp_re_4G, concat_df_5G, on=['地市', '告警类别'])
        concat_df_45G.to_csv(os.path.join(alert_to_base, time_key, time_key + '_bs_stati.csv'), index=False, encoding='utf_8')
        # concat_df_45G.to_csv(os.path.join('/data/resources/result_stati', time_key + '_bs_stati.csv'), index=False, encoding='utf_8')
        file_path = os.path.join(alert_to_base, time_key, time_key + '_bs_stati.csv')
        cmd = 'scp {} root@82.221.9.85:{}'.format(file_path, '/data/resources/result_stati/')
        print(cmd)
        os.system(cmd)


def alert_sum_stat_with_city(time_key, alert_num, warn_table_dir, alert_all_base, alert_noclear_base, alert_to_base,
                             Alert_all_convert_base, Alert_no_clear_convert_base, group='4G'):
    class_mapping, alert_class_list = get_warn_class_map(warn_table_dir, group)
    select_cols = ['地市', '已消除告警数', '未消除告警数', '总计']
    rename_dict = {'已消除告警数': group + '_已消除告警数', '未消除告警数': group + '_未消除告警数', '总计': group + '_总计'}
    cls_dict = {}
    for cls in alert_class_list:
        cls_dict[cls] = pd.DataFrame([], columns=['地市', '厂家', '已消除告警数', '未消除告警数', '总计'])
    print('遍历所有告警文件')
    for root, dirs, files in tqdm(os.walk(alert_all_base)):
        for file in files:
            if time_key in file:
                tmp_dict = {'地市': 0, '厂家': 0, '已消除告警数': 0, '未消除告警数': 0, '总计': 0}
                # print(root,file)
                file_split = file.split('_')
                city = file_split[0]
                mfrs = file_split[1]
                # if group=='5G' and mfrs in ['爱立信', '诺基亚']:
                if group == '5G' and mfrs in ['诺基亚']:
                    continue
                if not os.path.exists(root.replace(alert_all_base, Alert_all_convert_base)):
                    os.makedirs(root.replace(alert_all_base, Alert_all_convert_base))
                warn_class = stats_group_convert(root, root.replace(alert_all_base, Alert_all_convert_base), file, mfrs,
                                                 class_mapping)
                num_alert = (warn_class[list(set(class_mapping[mfrs].values()))].fillna(0)).astype(int).sum(axis=0)
                for cls in num_alert.index:
                    tmp_dict['地市'] = city
                    tmp_dict['厂家'] = mfrs
                    tmp_dict['总计'] = num_alert[cls]
                    cls_dict[cls] = cls_dict[cls].append(tmp_dict, ignore_index=True)
    print('遍历未消除告警文件')
    for root, dirs, files in tqdm(os.walk(alert_noclear_base)):
        for file in files:
            if time_key in file:
                # print(root,file)
                file_split = file.split('_')
                city = file_split[0]
                mfrs = file_split[1]
                # if group=='5G' and mfrs in ['爱立信', '诺基亚']:
                if group == '5G' and mfrs in ['诺基亚']:
                    continue
                if not os.path.exists(root.replace(alert_noclear_base, Alert_no_clear_convert_base)):
                    os.makedirs(root.replace(alert_noclear_base, Alert_no_clear_convert_base))
                warn_class = stats_group_convert(root, root.replace(alert_noclear_base, Alert_no_clear_convert_base),
                                                 file, mfrs, class_mapping)
                num_alert = (warn_class[list(set(class_mapping[mfrs].values()))].fillna(0)).astype(int).sum(axis=0)
                for cls in num_alert.index:
                    cls_dict[cls].loc[(cls_dict[cls]['地市'] == city) & (cls_dict[cls]['厂家'] == mfrs), '未消除告警数'] = \
                        num_alert[cls]

    concat_df = pd.DataFrame()
    concat_df_5G = pd.DataFrame()
    for cls in alert_class_list:
        if cls == 'OMC':
            print()
        cls_dict[cls]['已消除告警数'] = cls_dict[cls]['总计'] - cls_dict[cls]['未消除告警数']
        tmp_re = cls_dict[cls].groupby('地市').sum()
        tmp_re.reset_index(inplace=True)
        all_city_sum = tmp_re[tmp_re['地市'] == '汇总']
        all_city_sum.loc[:, '地市'] = '全省'
        tmp_re.drop(tmp_re[tmp_re['地市'] == '汇总'].index, axis=0, inplace=True)
        tmp_re = tmp_re.append(all_city_sum, ignore_index=True)
        tmp_re = tmp_re[select_cols].rename(columns=rename_dict)
        if group == '4G':
            if not os.path.exists(os.path.join(alert_to_base, time_key)):
                os.makedirs(os.path.join(alert_to_base, time_key))
            # tmp_re.to_csv(
            #     os.path.join(alert_to_base,time_key, '告警数统计', time_key + '_' + cls + '_告警数统计.csv'), index=False,
            #     encoding='utf_8_sig')
            tmp_re['告警类别'] = cls
            concat_df = pd.concat([concat_df, tmp_re])

        if group == '5G':
            tmp_re['告警类别'] = cls
            concat_df_5G = pd.concat([concat_df_5G, tmp_re])
    if group == '4G':
        analyse_start_date, analyse_end_date = get_current_week()
        concat_df['统计周期开始时间'] = str(analyse_start_date)
        concat_df['统计周期结束时间'] = str(analyse_end_date)
        concat_df = concat_df[['统计周期开始时间', '统计周期结束时间', '地市', '告警类别', '4G_已消除告警数', '4G_未消除告警数', '4G_总计']]

        concat_df.to_csv(os.path.join(alert_to_base, time_key, time_key + '_alarm_stati.csv'), index=False,
                         encoding='utf_8')
        file_path = os.path.join(alert_to_base, time_key, time_key + '_alarm_stati.csv')
        cmd = 'scp {} root@82.221.9.85:{}'.format(file_path, '/data/resources/result_stati/')
        print(cmd)
        os.system(cmd)
    if group == '5G':
        tmp_re_4G = pd.read_csv(os.path.join(alert_to_base, time_key, time_key + '_alarm_stati.csv'),
                                engine='python', encoding='utf_8')
        concat_df_5G = concat_df_5G[['地市', '告警类别', '5G_已消除告警数', '5G_未消除告警数', '5G_总计']]
        concat_df_45G = pd.merge(tmp_re_4G, concat_df_5G, on=['地市', '告警类别'])
        concat_df_45G.to_csv(
            os.path.join(alert_to_base, time_key, time_key + '_alarm_stati.csv'), index=False,
            encoding='utf_8')
        # concat_df_45G.to_csv(os.path.join('/data/resources/result_stati', time_key + '_alarm_stati.csv'), index=False, encoding='utf_8')
        file_path = os.path.join(alert_to_base, time_key, time_key + '_alarm_stati.csv')
        cmd = 'scp {} root@82.221.9.85:{}'.format(file_path, '/data/resources/result_stati/')
        print(cmd)
        os.system(cmd)


def alert_sum_stat_with_city1(time_key, ftype, warn_table_dir, alert_all_base, alert_noclear_base, alert_to_base,
                             Alert_all_convert_base, Alert_no_clear_convert_base, group='4G'):
    class_mapping, alert_class_list = get_warn_class_map(warn_table_dir, group)
    select_cols = ['地市', '已消除告警数', '未消除告警数', '总计']
    rename_dict = {'已消除告警数': group + '_已消除告警数', '未消除告警数': group + '_未消除告警数', '总计': group + '_总计'}
    cls_dict = {}
    for cls in alert_class_list:
        cls_dict[cls] = pd.DataFrame([], columns=['地市', '厂家', '已消除告警数', '未消除告警数', '总计'])
    print('遍历所有告警文件')
    for root, dirs, files in tqdm(os.walk(alert_all_base)):
        for file in files:
            if time_key in file:
                if ftype in file:
                    tmp_dict = {'地市': 0, '厂家': 0, '已消除告警数': 0, '未消除告警数': 0, '总计': 0}
                    # print(root,file)
                    file_split = file.split('_')
                    city = file_split[0]
                    mfrs = file_split[1]
                    # if group=='5G' and mfrs in ['爱立信', '诺基亚']:
                    if group == '5G' and mfrs in ['诺基亚']:
                        continue
                    if not os.path.exists(root.replace(alert_all_base, Alert_all_convert_base)):
                        os.makedirs(root.replace(alert_all_base, Alert_all_convert_base))
                    warn_class = stats_group_convert(root, root.replace(alert_all_base, Alert_all_convert_base), file, mfrs,
                                                     class_mapping)
                    num_alert = (warn_class[list(set(class_mapping[mfrs].values()))].fillna(0)).astype(int).sum(axis=0)
                    for cls in num_alert.index:
                        tmp_dict['地市'] = city
                        tmp_dict['厂家'] = mfrs
                        tmp_dict['总计'] = num_alert[cls]
                        cls_dict[cls] = cls_dict[cls].append(tmp_dict, ignore_index=True)
    print('遍历未消除告警文件')
    for root, dirs, files in tqdm(os.walk(alert_noclear_base)):
        for file in files:
            if time_key in file:
                if ftype in file:
                    # print(root,file)
                    file_split = file.split('_')
                    city = file_split[0]
                    mfrs = file_split[1]
                    # if group=='5G' and mfrs in ['爱立信', '诺基亚']:
                    if group == '5G' and mfrs in ['诺基亚']:
                        continue
                    if not os.path.exists(root.replace(alert_noclear_base, Alert_no_clear_convert_base)):
                        os.makedirs(root.replace(alert_noclear_base, Alert_no_clear_convert_base))
                    warn_class = stats_group_convert(root, root.replace(alert_noclear_base, Alert_no_clear_convert_base),
                                                     file, mfrs, class_mapping)
                    num_alert = (warn_class[list(set(class_mapping[mfrs].values()))].fillna(0)).astype(int).sum(axis=0)
                    for cls in num_alert.index:
                        cls_dict[cls].loc[(cls_dict[cls]['地市'] == city) & (cls_dict[cls]['厂家'] == mfrs), '未消除告警数'] = \
                            num_alert[cls]

    concat_df = pd.DataFrame()
    concat_df_5G = pd.DataFrame()
    for cls in alert_class_list:
        if cls == 'OMC':
            print()
        cls_dict[cls]['已消除告警数'] = cls_dict[cls]['总计'] - cls_dict[cls]['未消除告警数']
        tmp_re = cls_dict[cls].groupby('地市').sum()
        tmp_re.reset_index(inplace=True)
        all_city_sum = tmp_re[tmp_re['地市'] == '汇总']
        all_city_sum.loc[:, '地市'] = '全省'
        tmp_re.drop(tmp_re[tmp_re['地市'] == '汇总'].index, axis=0, inplace=True)
        tmp_re = tmp_re.append(all_city_sum, ignore_index=True)
        tmp_re = tmp_re[select_cols].rename(columns=rename_dict)
        if group == '4G':
            if not os.path.exists(os.path.join(alert_to_base, time_key)):
                os.makedirs(os.path.join(alert_to_base, time_key))
            # tmp_re.to_csv(
            #     os.path.join(alert_to_base,time_key, '告警数统计', time_key + '_' + cls + '_告警数统计.csv'), index=False,
            #     encoding='utf_8_sig')
            tmp_re['告警类别'] = cls
            concat_df = pd.concat([concat_df, tmp_re])

        if group == '5G':
            tmp_re['告警类别'] = cls
            concat_df_5G = pd.concat([concat_df_5G, tmp_re])
    if group == '4G':
        analyse_start_date, analyse_end_date = get_current_week()
        concat_df['统计周期开始时间'] = str(analyse_start_date)
        concat_df['统计周期结束时间'] = str(analyse_end_date)
        concat_df = concat_df[['统计周期开始时间', '统计周期结束时间', '地市', '告警类别', '4G_已消除告警数', '4G_未消除告警数', '4G_总计']]

        concat_df.to_csv(os.path.join(alert_to_base, time_key, time_key + '_alarm_stati_{}.csv'.format(ftype)), index=False,
                         encoding='utf_8')
    if group == '5G':
        tmp_re_4G = pd.read_csv(os.path.join(alert_to_base, time_key, time_key + '_alarm_stati_{}.csv'.format(ftype)),
                                engine='python', encoding='utf_8')
        concat_df_5G = concat_df_5G[['地市', '告警类别', '5G_已消除告警数', '5G_未消除告警数', '5G_总计']]
        concat_df_45G = pd.merge(tmp_re_4G, concat_df_5G, on=['地市', '告警类别'])
        concat_df_45G.to_csv(
            os.path.join(alert_to_base, time_key, time_key + '_alarm_stati.csv_{}'.format(ftype)), index=False,
            encoding='utf_8')




def get_current_week():
    monday, sunday = datetime.date.today(), datetime.date.today()
    one_day = datetime.timedelta(days=1)
    while monday.weekday() != 0:
        monday -= one_day
    while sunday.weekday() != 6:
        sunday += one_day
    monday, sunday = monday - datetime.timedelta(days=7), sunday - datetime.timedelta(days=7)
    return monday, sunday

def get_next_week():
    monday, sunday = datetime.date.today(), datetime.date.today()
    one_day = datetime.timedelta(days=1)
    while monday.weekday() != 0:
        monday -= one_day
    while sunday.weekday() != 6:
        sunday += one_day
    return monday, sunday

def get_check_week():
    def get_check_date(analyse_start_date, analyse_end_date):

        analyse_date_list1 = [str(date).split()[0].replace('-', '') for date in
                              pd.date_range(analyse_start_date, analyse_end_date)]
        return analyse_date_list1[0] + '-' + analyse_date_list1[-1][-4:]
    monday, sunday = datetime.date.today(), datetime.date.today()
    one_day = datetime.timedelta(days=1)
    while monday.weekday() != 0:
        monday -= one_day
    while sunday.weekday() != 6:
        sunday += one_day
    check_monday, check_sunday = monday - datetime.timedelta(days=7*5), sunday - datetime.timedelta(days=7*5)
    check_monday_1, check_sunday_1 = monday - datetime.timedelta(days=7*3), sunday - datetime.timedelta(days=7 * 3)
    check_monday_2, check_sunday_2 = monday - datetime.timedelta(days=7 * 2), sunday - datetime.timedelta(days=7 * 2)
    check_monday_3, check_sunday_3 = monday - datetime.timedelta(days=7), sunday - datetime.timedelta(days=7)
    check_monday_4, check_sunday_4 = monday, sunday
    check_date = get_check_date(check_monday, check_sunday)
    check_date_1 = get_check_date(check_monday_1, check_sunday_1)
    check_date_2 = get_check_date(check_monday_2, check_sunday_2)
    check_date_3 = get_check_date(check_monday_3, check_sunday_3)
    check_date_4 = get_check_date(check_monday_4, check_sunday_4)
    verify_date_list = [check_date_1, check_date_2, check_date_3, check_date_4]
    return check_date, verify_date_list


def time_convert(x):
    if str(x) != 'nan':
        try:
            str_h = re.findall(r'\d+', str(x).split(' ')[0])[0]
        except:
            print(x)
        str_m = re.findall(r'\d+', str(x).split(' ')[1])[0]
        str_m2h = int(str_h) + (int(str_m) / 60)
        return str_m2h  # 时间格式-小时
        # return x/60  # 分钟-小时

    else:
        return 0

def stati_province(data_metrics_city):
    data_metrics_province = pd.DataFrame(columns=['统计周期开始时间', '统计周期结束时间', '地市', '评估指标', '全量', '退服类', '厂家'])
    metrics_list = data_metrics_city['评估指标'].unique().tolist()
    for _metric in metrics_list:
        data_metrics_province_tmp = []
        data_metrics_city_tmp = data_metrics_city[data_metrics_city['评估指标']==_metric]
        data_metrics_city_tmp.fillna(0, inplace=True)
        data_metrics_province_tmp.append(data_metrics_city_tmp['统计周期开始时间'].unique()[0])
        data_metrics_province_tmp.append(data_metrics_city_tmp['统计周期结束时间'].unique()[0])
        data_metrics_province_tmp.append('全省')
        data_metrics_province_tmp.append(_metric)
        if str(_metric).split('_')[1] in ['告警量', '工单量', '工单处理总时长', '隐患基站总数量']:
            data_metrics_province_tmp.append(data_metrics_city_tmp['全量'].sum())
            data_metrics_province_tmp.append(data_metrics_city_tmp['退服类'].sum())
            print()
        elif str(_metric).split('_')[1] in ['工单处理平均时长', '隐患基站根治率']:
            data_metrics_province_tmp.append(data_metrics_city_tmp['全量'].mean())
            data_metrics_province_tmp.append(data_metrics_city_tmp['退服类'].mean())
            print()
        elif str(_metric).split('_')[1] in ['千基站隐患发现数量']:
            data_metrics_city_tmp = data_metrics_city[data_metrics_city['评估指标'] == str(_metric).split('_')[0]+'_隐患基站总数量']
            if str(_metric).split('_')[0] == '4G':
                data_nrm = pd.read_csv('/data/nrm/nrm_4G.csv')
            else:
                data_nrm = pd.read_csv('/data/nrm/nrm_5G.csv')
            data_metrics_province_tmp.append((data_metrics_city_tmp['全量'].sum())/(len(data_nrm))*1000)
            data_metrics_province_tmp.append((data_metrics_city_tmp['退服类'].sum())/(len(data_nrm))*1000)
            # data_metrics_province_tmp.append((data_metrics_city_tmp['全量'].sum()) / (200000)*1000)
            # data_metrics_province_tmp.append((data_metrics_city_tmp['退服类'].sum()) / (200000)*1000)
        data_metrics_province.loc[len(data_metrics_province)] = data_metrics_province_tmp
        print()
    data_metrics_all = pd.concat([data_metrics_city, data_metrics_province], ignore_index=True)
    return data_metrics_all




def get_evaluate_indicator(data_alarm, start, end, year, wo_dir, evaluate_type, district, Warn_table_dir, date_toplist, ftype, gc='4G'):
    """
    :param data_alarm: 告警统计数据
    :param start: 开始日期
    :param end: 结束日期
    :param year: 年份
    :param wo_dir: 工单目录
    :param evaluate_type: 评估范围：全地市、退服类
    :param district:  地市
    :param Warn_table_dir: 告警分类表，可读退服告警标题
    :param gc: 制式
    :return: 告警量 工单量、工单处理时长、工单处理平均时长
    """
    date = year + start + '-' + end
    start_date = datetime.datetime(int(year), int(start[:2]), int(start[-2:]))
    # start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
    if gc == '4G':
        enodebid_ = 'ENODEBID'
    else:
        enodebid_ = 'ENODEBID'
    delta = datetime.timedelta(days=7)
    end_date = start_date + delta
    # end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")
    if evaluate_type == '全量':
        alarm_num = sum(data_alarm[data_alarm['地市'].str.contains(district)][gc + '_总计'].values.tolist())
    elif evaluate_type == '退服类':
        data_alarm = data_alarm[data_alarm['地市'].str.contains(district)]
        print(data_alarm[data_alarm['告警类别'].isin(['退服', '托管和退服'])][gc + '_总计'].values)
        try:
            alarm_num = data_alarm[data_alarm['告警类别'].isin(['退服', '托管和退服'])][gc + '_总计'].values[0]
        except:
            alarm_num = 0
    df_concat = pd.read_csv(open(os.path.join(wo_dir, 'wo_' + date + '.csv')))
    df_concat.dropna(subset=['发生地区'], inplace=True)
    # print(df_concat['发生地区'].unique().tolist())
    df_concat = df_concat[df_concat['发生地区'].str.contains(district)]
    df_concat = df_concat[df_concat['故障设备厂商'].str.contains(ftype)]
    if gc == '4G':
        df_concat = df_concat[df_concat['故障设备类型'].isin(['eNodeB', 'EUTRANCELL'])]
    if gc == '5G':
        df_concat = df_concat[df_concat['故障设备类型'].isin(['GNodeB', 'Gcell'])]
    df_concat['发生时间'] = pd.to_datetime(df_concat['发生时间'])
    df_concat = df_concat[(df_concat['发生时间'] >= start_date) & (df_concat['发生时间'] < end_date)]
    df_concat.sort_values(by='发生时间', inplace=True)
    if evaluate_type == '全量':
        df_concat = df_concat
    elif evaluate_type == '退服类':
        if gc == '4G':
            df_alarm_class = pd.read_excel(Warn_table_dir, sheet_name='4G')
        else:
            df_alarm_class = pd.read_excel(Warn_table_dir, sheet_name='5G')
        df_alarm_class.dropna(subset=['专题大类'], inplace=True)
        df_alarm_class = df_alarm_class[['厂家', '告警标题', '专题大类']]
        tf_list = df_alarm_class[df_alarm_class['专题大类'].isin(['退服', '脱管和退服'])]['告警标题'].tolist()
        tf_list = '|'.join(tf_list)
        tf_list_1 = "退服|基站退服|小区不可用告警|同局站基站退服|小区退服|网元断链告警|CELL FAULTY"  # 需要确定退服类告警种类
        tf_list = '|'.join([tf_list_1, tf_list])
        df_concat = df_concat[df_concat['工单标题'].str.contains(tf_list)]
    wo_num = len(df_concat)
    # df_concat['历时'] = df_concat['历时'].apply(lambda x: time_convert(x))  # 新疆的历时是分钟，转变成小时
    df_concat['历时'] = ((pd.to_datetime(df_concat['故障消除时间']) - pd.to_datetime(df_concat['发生时间'])) / pd.Timedelta(1,'H')).fillna(0)
    wo_faultdurattime = sum([0 if x == ' ' else float(x) for x in df_concat['历时'].values.tolist()])
    if wo_num != 0:
        wo_faultdurattime_average = wo_faultdurattime / wo_num
    else:
        wo_faultdurattime_average = 0
    """
    新增统计指标：隐患基站总数量，千基站隐患发现数量，隐患基站根治率
    """
    ######隐患基站总数量

    if gc == '4G':
        abs_path = '/data/resources/result/'
        data_gol = pd.DataFrame()
        # if city_ not in ['奎屯']:
        for filename_ in os.listdir(abs_path):
            if date_toplist in filename_:
                if 'Top异常基站清单' in filename_:
                    if district in filename_:
                        if '4G' in filename_:
                            if ftype in filename_:
                                data_gol_ = pd.read_csv(os.path.join(abs_path, filename_), sep='\$\$')
                                data_gol = pd.concat([data_gol, data_gol_], ignore_index=True)
                                data_gol = data_gol[data_gol['制式'].isin(['4G'])]
                            print('{}当周{} TOP隐患基站总数：'.format(district, gc), len(data_gol))
    elif gc == '5G':
        abs_path = '/data/resources/result/'
        data_gol = pd.DataFrame()
        # if city_ not in ['克拉玛依','阿克苏','石河子','塔城','博州']:
        for filename_ in os.listdir(abs_path):
            if date_toplist in filename_:
                if 'Top异常基站清单' in filename_:
                    if district in filename_:
                        if '5G' in filename_:
                            if ftype in filename_:
                                data_gol_ = pd.read_csv(os.path.join(abs_path, filename_), sep='\$\$')
                                data_gol = pd.concat([data_gol, data_gol_], ignore_index=True)
                                data_gol = data_gol[data_gol['制式'].isin(['5G'])]
                            print('{}当周{} TOP隐患基站总数：'.format(district, gc), len(data_gol))
    else:
        print('!!!Error')
    top_bts_num = len(data_gol)

    ######千基站隐患发现数量
    if gc == '4G':
        data_nrm = pd.read_csv('/data/nrm/nrm_4G.csv')
        # data_nrm = data_nrm[data_nrm['所属地市'].isin([city_ + '市'])]
        data_nrm = data_nrm[data_nrm['所属地市'].isin([district])]
    else:
        data_nrm = pd.read_csv('/data/nrm/nrm_5G.csv')
        data_nrm = data_nrm[data_nrm['所属地市'].isin([district])]
    all_bts_num = len(data_nrm)
    top_bts_num_perThousand = top_bts_num/all_bts_num*1000

    ######隐患基站根治率
    check_date, verify_date_list = get_check_week()
    if gc == '4G':
        abs_path = '/data/resources/result/'
        data_gol = pd.DataFrame()
        # if city_ not in ['奎屯']:
        for filename_ in os.listdir(abs_path):
            if check_date in filename_:
                if 'Top异常基站清单' in filename_:
                    if district in filename_:
                        if '4G' in filename_:
                            if ftype in filename_:
                                data_gol_ = pd.read_csv(os.path.join(abs_path, filename_), sep='\$\$')
                                data_gol = pd.concat([data_gol, data_gol_], ignore_index=True)
    elif gc == '5G':
        abs_path = '/data/resources/result/'
        data_gol = pd.DataFrame()
        # if city_ not in ['克拉玛依','阿克苏','石河子','塔城','博州']:
        for filename_ in os.listdir(abs_path):
            if check_date in filename_:
                if 'Top异常基站清单' in filename_:
                    if district in filename_:
                        if '5G' in filename_:
                            if ftype in filename_:
                                data_gol_ = pd.read_csv(os.path.join(abs_path, filename_), sep='\$\$')
                                data_gol = pd.concat([data_gol, data_gol_], ignore_index=True)
    else:
        print('!!!Error')
    if len(data_gol) == 0:
        print('当是未输出当周TOP隐患基站清单！！！')
        radical_rate = np.nan
    else:
        check_bts_list = data_gol[enodebid_].tolist()
        if gc=='4G':
            abs_path = '/data/resources/result/'
        else:
            abs_path = '/data/resources/result/'
        data_gol_verify = pd.DataFrame()
        # if gc=='5G' and city_ not in ['克拉玛依','阿克苏','石河子','塔城','博州']:
        if gc=='5G':
            for filename_ in os.listdir(abs_path):
                if filename_.split('_')[-1].split('.')[0] in verify_date_list:
                    if 'Top异常基站清单' in filename_:
                        if district in filename_:
                            if '5G' in filename_:
                                if ftype in filename_:
                                    data_gol_ = pd.read_csv(os.path.join(abs_path, filename_), sep='\$\$')
                                    data_gol_verify = pd.concat([data_gol_verify, data_gol_], ignore_index=True)
        # elif gc=='4G' and city_ not in ['奎屯']:
        elif gc=='4G':
            for filename_ in os.listdir(abs_path):
                if filename_.split('_')[-1].split('.')[0] in verify_date_list:
                    if 'Top异常基站清单' in filename_:
                        if district in filename_:
                            if '4G' in filename_:
                                if ftype in filename_:
                                    data_gol_ = pd.read_csv(os.path.join(abs_path, filename_), sep='\$\$')
                                    data_gol_verify = pd.concat([data_gol_verify, data_gol_], ignore_index=True)
                                    data_gol_verify = data_gol_verify[data_gol_verify[enodebid_].isin(check_bts_list)]
        if len(data_gol_verify)==0:  ## 这个地方会有些问题，当是5G丽水时，根治率为1,其实丽水没有5G，是找不到文件
            radical_rate = 1
        else:
            not_radical_bts = data_gol_verify[enodebid_].unique().tolist()
            radical_rate = (len(check_bts_list)-len(not_radical_bts))/len(check_bts_list)

    return alarm_num, wo_num, wo_faultdurattime, wo_faultdurattime_average,\
           top_bts_num, top_bts_num_perThousand, radical_rate


def main():
    # param_con.parser.add_argument('--time_key', type=str, default=mode)
    start_time = time.clock()
    param = param_con.params()
    Warn_table_dir = param.Warn_table_dir  # 告警分类表路径，可固定不变
    Alert_to_base = param.Alert_to_base  # 统计结果 目录，可固定不变
    alert_num_path = param.alert_num_path  # 总基站数文件 路径，可固定不变
    print('统计开始时间：{}'.format(time.ctime()))
    # wuxian_base_dir = param.wuxian_base_dir    #4G无线原始数据目录,可固定不变
    orgpath = param.orgpath  # 中间数据根目录
    analyse_start_date, analyse_end_date = get_current_week()
    analyse_date_list = [str(date).split()[0].replace('-', '') for date in
                         pd.date_range(analyse_start_date, analyse_end_date)]
    time_key = analyse_date_list[0] + '-' + analyse_date_list[-1][-4:]
    analyse_start_date1, analyse_end_date1 = get_current_week()
    analyse_date_list1 = [str(date).split()[0].replace('-', '') for date in
                         pd.date_range(analyse_start_date1, analyse_end_date1)]
    date_toplist = analyse_date_list1[0] + '-' + analyse_date_list1[-1][-4:]
    year = time_key[:4]
    start = time_key.split('-')[0][4:]
    end = time_key.split('-')[1]
    wuxian_base_dir = '/data/alarm/{}/enodeb'.format(year + start + '-' + end)
    wuxian_base_dir_5G = '/data/alarm/{}/gnodeb'.format(year + start + '-' + end)
    Alert_all_base_4G = os.path.join(orgpath, 'AIOps_all_4G', 'Alert_Samp')
    Alert_all_convert_base_4G = os.path.join(orgpath, 'AIOps_all_4G', 'Alert_Samp_convert')
    Alert_no_clear_base_4G = os.path.join(orgpath, 'AIOps_no_clear_4G', 'Alert_Samp')
    Alert_no_clear_convert_base_4G = os.path.join(orgpath, 'AIOps_no_clear_4G', 'Alert_Samp_convert')
    Alert_all_base_5G = os.path.join(orgpath, 'AIOps_all_5G', 'Alert_Samp')
    Alert_all_convert_base_5G = os.path.join(orgpath, 'AIOps_all_5G', 'Alert_Samp_convert')
    Alert_no_clear_base_5G = os.path.join(orgpath, 'AIOps_no_clear_5G', 'Alert_Samp')
    Alert_no_clear_convert_base_5G = os.path.join(orgpath, 'AIOps_no_clear_5G', 'Alert_Samp_convert')
    excel_Alert_data_convert(wuxian_base_dir=wuxian_base_dir, orgpath=orgpath, start=start, end=end, year=year,
                             gc='4G')
    excel_Alert_data_convert(wuxian_base_dir=wuxian_base_dir_5G, orgpath=orgpath, start=start, end=end, year=year, gc='5G')
    dict_4G, dict_5G = read_node_num(alert_num_path)
    print('统计4G告警开始时间：{}'.format(time.ctime()))
    alert_stat_with_city(time_key, dict_4G, Warn_table_dir, Alert_all_base_4G, Alert_no_clear_base_4G,
                         Alert_to_base,
                         Alert_all_convert_base_4G, Alert_no_clear_convert_base_4G, '4G')
    # 上面统计分专题的基站数，下面统计分专题的告警数
    alert_sum_stat_with_city(time_key, dict_4G, Warn_table_dir, Alert_all_base_4G, Alert_no_clear_base_4G,
                             Alert_to_base,
                             Alert_all_convert_base_4G, Alert_no_clear_convert_base_4G, '4G')
    print('统计5G告警开始时间：{}'.format(time.ctime()))
    alert_stat_with_city(time_key, dict_5G, Warn_table_dir, Alert_all_base_5G, Alert_no_clear_base_5G, Alert_to_base,
                         Alert_all_convert_base_5G, Alert_no_clear_convert_base_5G, '5G')
    # 上面统计分专题的基站数，下面统计分专题的告警数
    alert_sum_stat_with_city(time_key, dict_5G, Warn_table_dir, Alert_all_base_5G, Alert_no_clear_base_5G, Alert_to_base, Alert_all_convert_base_5G, Alert_no_clear_convert_base_5G, '5G')
    print('统计结束时间：{}'.format(time.ctime()))
    print('统计执行完成 in {} seconds'.format(time.clock() - start_time))

    """
    统计相关评估指标：工单量、工单处理时长、工单处理平均时长、隐患基站总数量、千基站隐患发现数量、隐患基站根治率
    （区分全地市和退服类）
    """
    wo_dir = param.wo_dir
    evaluate_types = ['全量', '退服类']
    data_evaluate_all = pd.DataFrame()
    data_evaluate = pd.DataFrame(columns=['统计周期开始时间', '统计周期结束时间', '地市', '全量', '退服类', '厂家'],
                                 index=['4G_告警量', '4G_工单量', '4G_工单处理总时长', '4G_工单处理平均时长',
                                        '4G_隐患基站总数量', '4G_千基站隐患发现数量', '4G_隐患基站根治率'])
    districts = ['合肥','芜湖','蚌埠','淮南','马鞍山','淮北','铜陵','安庆','黄山','滁州','阜阳','宿州','六安','亳州','池州','宣城']
    ftypes = ['华为', '中兴', '诺基亚', '爱立信']
    # data_alarm = pd.read_csv(
    #     open(os.path.join(Alert_to_base, time_key, time_key + '_alarm_stati.csv'), encoding='utf-8'))
    for dist_ in districts:
        for ftype in ftypes:
            for evaluate_type in evaluate_types:
                alert_sum_stat_with_city1(time_key, ftype, Warn_table_dir, Alert_all_base_4G,
                                         Alert_no_clear_base_4G,
                                         Alert_to_base,
                                         Alert_all_convert_base_4G, Alert_no_clear_convert_base_4G, '4G')
                alert_sum_stat_with_city1(time_key, ftype, Warn_table_dir, Alert_all_base_5G,
                                         Alert_no_clear_base_5G, Alert_to_base, Alert_all_convert_base_5G,
                                         Alert_no_clear_convert_base_5G, '5G')
                data_alarm = pd.read_csv(
                    open(os.path.join(Alert_to_base, time_key, time_key + '_alarm_stati_{}.csv'.format(ftype)), encoding='utf-8'))
                alarm_num, wo_num, wo_faultdurattime, wo_faultdurattime_average, top_bts_num, top_bts_num_perThousand, radical_rate = get_evaluate_indicator(
                    data_alarm=data_alarm, start=start, end=end, year=year, wo_dir=wo_dir, evaluate_type=evaluate_type,
                    district=dist_, Warn_table_dir=Warn_table_dir, date_toplist=date_toplist, ftype = ftype, gc='4G')
                eva_list = [alarm_num, wo_num, wo_faultdurattime, wo_faultdurattime_average, top_bts_num, top_bts_num_perThousand, radical_rate]
                data_evaluate[evaluate_type] = eva_list
                data_evaluate['地市'] = dist_
                data_evaluate['统计周期开始时间'] = str(analyse_start_date)
                data_evaluate['统计周期结束时间'] = str(analyse_end_date)
                data_evaluate['地市'] = dist_
                data_evaluate['厂家'] = ftype
            data_evaluate_all = pd.concat([data_evaluate_all, data_evaluate])
    data_evaluate_all.reset_index(drop=False, inplace=True)
    data_evaluate_all.rename(columns={'index': '评估指标'}, inplace=True)
    data_evaluate_all = data_evaluate_all[['统计周期开始时间', '统计周期结束时间', '地市', '评估指标', '全量', '退服类', '厂家']]
    data_evaluate_all_new = stati_province(data_evaluate_all)
    # 统计全省top清单基站数，隐患根治率，千基站数
    if not os.path.exists(os.path.join(Alert_to_base, time_key)):
        os.makedirs(os.path.join(Alert_to_base, time_key))
    data_evaluate_all_new.to_csv(os.path.join(Alert_to_base, time_key, time_key + '_metrics_stati_4G.csv'), index=False,
                             encoding='utf_8')
    # data_evaluate_all.to_csv(os.path.join('/data/resources/result_stati', time_key + '_metrics_stati.csv'),index=False,encoding='utf_8')
    file_path = os.path.join(Alert_to_base, time_key, time_key + '_metrics_stati_4G.csv')
    cmd = 'scp {} root@82.221.9.85:{}'.format(file_path, '/data/resources/result_stati/')
    print(cmd)
    os.system(cmd)

    data_evaluate_all = pd.DataFrame()
    data_evaluate = pd.DataFrame(columns=['统计周期开始时间', '统计周期结束时间', '地市', '全量', '退服类', '厂家'],
                                 index=['5G_告警量', '5G_工单量', '5G_工单处理总时长', '5G_工单处理平均时长',
                                        '5G_隐患基站总数量', '5G_千基站隐患发现数量', '5G_隐患基站根治率'])
    # 统计5G指标
    for dist_ in districts:
        for ftype in ftypes:
            for evaluate_type in evaluate_types:
                alarm_num, wo_num, wo_faultdurattime, wo_faultdurattime_average, top_bts_num, top_bts_num_perThousand, radical_rate = get_evaluate_indicator(
                    data_alarm=data_alarm, start=start, end=end, year=year, wo_dir=wo_dir, evaluate_type=evaluate_type,
                    district=dist_, Warn_table_dir=Warn_table_dir, date_toplist=date_toplist, ftype = ftype, gc='5G')
                eva_list = [alarm_num, wo_num, wo_faultdurattime, wo_faultdurattime_average, top_bts_num, top_bts_num_perThousand, radical_rate]
                data_evaluate[evaluate_type] = eva_list
                data_evaluate['地市'] = dist_
                data_evaluate['统计周期开始时间'] = str(analyse_start_date)
                data_evaluate['统计周期结束时间'] = str(analyse_end_date)
                data_evaluate['地市'] = dist_
                data_evaluate['厂家'] = ftype
            data_evaluate_all = pd.concat([data_evaluate_all, data_evaluate])
    data_evaluate_all.reset_index(drop=False, inplace=True)
    data_evaluate_all.rename(columns={'index': '评估指标'}, inplace=True)
    data_evaluate_all = data_evaluate_all[['统计周期开始时间', '统计周期结束时间', '地市', '评估指标', '全量', '退服类', '厂家']]
    # 与4G的合并
    # data_evaluate_all_4G = pd.read_csv(os.path.join(Alert_to_base, time_key, time_key + '_metrics_stati.csv'))
    # data_evaluate_all_new = pd.concat([data_evaluate_all_4G, data_evaluate_all], ignore_index=True)
    data_evaluate_all_new = stati_province(data_evaluate_all)
    if not os.path.exists(os.path.join(Alert_to_base, time_key)):
        os.makedirs(os.path.join(Alert_to_base, time_key))
    data_evaluate_all_new.to_csv(os.path.join(Alert_to_base, time_key, time_key + '_metrics_stati_5G.csv'), index=False,
                                 encoding='utf_8')
    # data_evaluate_all_new.to_csv(os.path.join('/data/resources/result_stati', time_key + '_metrics_stati.csv'), index=False, encoding='utf_8')
    file_path = os.path.join(Alert_to_base, time_key, time_key + '_metrics_stati_5G.csv')
    cmd = 'scp {} root@82.221.9.85:{}'.format(file_path, '/data/resources/result_stati/')
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    '''
    args = sys.argv[1]
    args = args.split(',')
    time_key = args[0]
    alert_num = args[1]

    '''
    main()

