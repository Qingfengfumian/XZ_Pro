import numpy as np
import pandas as pd
import os
import re
import datetime
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv
import json
import csv
from utils import mkdir
from utils import real_warns_5G, real_warns_4G
# from utils import mysql_dealdata
mysql_dealdata = []

csv.field_size_limit(500 * 1024 * 1024)


def predict(para):
    distr_list = para.distr_list
    ftype = para.ftype
    date = para.date
    yy1 = int(date.split('-')[1][0:4])
    mm1 = int(date.split('-')[1][4:6])
    dd1 = int(date.split('-')[1][6:8])
    base_date = datetime.date(yy1, mm1, dd1)
    Inspect_date = datetime.datetime.strftime(base_date + datetime.timedelta(days=1), "%Y%m%d")

    for distr in distr_list:
        print(distr + '退服')
        for Tel_mode in ['4G', '5G']:
            distr_dict = {'拉萨':'LS','昌都':'CD','山南':'SN','林芝':'LZ','日喀则':'RKZ','那曲':'NQ','阿里':'AL'}
            distr_new = distr_dict.get(distr)
            try:
                XJ_res = pd.read_csv(r"./Data/{}/Inspect_List/{}/Origin_{}_{}.csv".format(distr, ftype, date, Tel_mode),
                                     encoding='gbk', engine='python')
            except:
                print("没有文件/Data/{}/Inspect_List/{}/Origin_{}_{}.csv".format(distr, ftype, date, Tel_mode))
                continue
            TF_res_o = pd.read_csv(r"./Data/{}/Inspect_List/{}/TFPre_{}_{}.csv".format(distr, ftype, date, Tel_mode),
                                   encoding='gbk', engine='python')
            TF_res_o['未来三天退服概率'] = 0.3 + 0.45 * TF_res_o['pred_probability_6D'] + 0.25 * TF_res_o['pred_probability_7D']
            # if distr in ['北京']:
            #     TF_res_o = TF_res_o.rename(columns={'pred_probability': '未来三天退服概率'})
            # elif distr in ['铜川','汉中','宝鸡', '咸阳','西安','商洛','安康', '渭南', '延安', '榆林']:  # 因为输出了多个退服概率，需要改一下列名，由于6D模型退服概率普遍低，加0.15，如果超过1的话要减一部分
            #     TF_res_o = TF_res_o.rename(columns={'未来三天退服概率': '未来三天退服概率1','pred_probability_6D': '未来三天退服概率'})
            #     TF_res_o['未来三天退服概率'] = TF_res_o['未来三天退服概率']+0.15
            #     TF_res_o['未来三天退服概率'] = TF_res_o['未来三天退服概率'].map(lambda x:1-0.054172392 if x>=1 else x)
            TF_res_o['基站id'] = TF_res_o['基站id'].map(lambda x: str(x).split('.')[0])
            try:
                gongcan = pd.read_csv(r"./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv", encoding='gbk', engine='python')
            except:
                gongcan = pd.read_csv(r"./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv", encoding='utf-8', engine='python')
            GongCan_W = gongcan[['基站名称', 'ENODEB_ID']]
            GongCan_W.rename(columns={'ENODEB_ID': '基础小区号'}, inplace=True)
            GongCan_W['基础小区号'] = GongCan_W['基础小区号'].map(lambda x: str(x).split('-')[0].split(".")[0])
            GongCan_W_del = GongCan_W.drop_duplicates(subset=['基础小区号', '基站名称'], keep='first')
            GongCan_W_del = GongCan_W_del.rename(columns={'基础小区号': '基站id', '基站名称': '所属基站'})
            TF_res = TF_res_o.copy()
            TF_res_GC = pd.merge(TF_res, GongCan_W_del, how='left')
            # ------------------------ 输出退服清单到服务器 ------------------
            if ftype == '华为':
                if Tel_mode == '4G':
                    if distr in ['昌都','山南','林芝']:  # 西藏4G华为 筛选 6D>=0.60 7D>=0.60 前7天退服天数>=1
                        try:
                            SY_Del6D_HW_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_SY_4G_HW', IF_multi=0)  # 0为一对多 1位一对一
                            SY_Del7D_HW_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_SY_4G_HW', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            SY_Del6D_HW_4G = 0.60  # 76%
                            SY_Del7D_HW_4G = 0.60
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= SY_Del6D_HW_4G) | (
                                TF_res_GC['pred_probability_7D'] >= SY_Del7D_HW_4G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        # TF_res_GC_select = TF_res_GC_select.rename(columns={'未来三天退服概率': '未来三天退服概率6D', '未来三天退服概率1': '未来三天退服概率'})
                        TF_res_GC_2 = TF_res_GC_select.copy()
                    elif distr in ['拉萨']:  # 拉萨4G华为 筛选 6D>=0.65 7D>=0.65 前7天退服天数>=1
                        try:
                            XY_Del6D_HW_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_XY_4G_HW', IF_multi=0)  # 0为一对多 1位一对一
                            XY_Del7D_HW_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_XY_4G_HW', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            XY_Del6D_HW_4G = 0.65  # 76%
                            XY_Del7D_HW_4G = 0.65
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= XY_Del6D_HW_4G) | (
                                TF_res_GC['pred_probability_7D'] >= XY_Del7D_HW_4G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()

                elif Tel_mode == '5G':
                    if distr in ['拉萨','昌都','山南','林芝']:  # 西藏5G华为 筛选 6D>=0.6 7D>=0.6 前7天退服天数>=1
                        try:
                            SY_Del6D_HW_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_SY_5G_HW', IF_multi=0)  # 0为一对多 1位一对一
                            SY_Del7D_HW_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_SY_5G_HW', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            SY_Del6D_HW_5G = 0.65  # 76%
                            SY_Del7D_HW_5G = 0.65
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= SY_Del6D_HW_5G) | (
                                TF_res_GC['pred_probability_7D'] >= SY_Del7D_HW_5G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()
                    elif distr in ['XX', 'XX']:  # 筛选 6D>=0.75 即'未来三天退服概率'=0.75 7D>=0.8 前7天退服天数>=1
                        try:
                            XY_Del6D_HW_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_XY_5G_HW', IF_multi=0)  # 0为一对多 1位一对一
                            XY_Del7D_HW_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_XY_5G_HW', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            XY_Del6D_HW_5G = 0.7
                            XY_Del7D_HW_5G = 0.8
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= XY_Del6D_HW_5G) | (
                                TF_res_GC['pred_probability_7D'] >= XY_Del7D_HW_5G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()

            elif ftype == '中兴':
                if Tel_mode == '4G':
                    if distr in ['日喀则','那曲','阿里']:  # 内蒙4G中兴 筛选 6D>=0.55 7D>=0.55 前7天退服天数>=1
                        try:
                            SY_Del6D_ZX_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_SY_4G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                            SY_Del7D_ZX_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_SY_4G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            SY_Del6D_ZX_4G = 0.55  # 79%
                            SY_Del7D_ZX_4G = 0.55
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= SY_Del6D_ZX_4G) | (
                                TF_res_GC['pred_probability_7D'] >= SY_Del7D_ZX_4G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()
                elif Tel_mode == '5G':
                    if distr in ['日喀则','那曲','阿里']:  # 内蒙5G中兴 筛选 6D>=0.5 7D>=0.5 前7天退服天数>=1
                        try:
                            SY_Del6D_ZX_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_SY_5G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                            SY_Del7D_ZX_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_SY_5G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            SY_Del6D_ZX_5G = 0.5  # 84%
                            SY_Del7D_ZX_5G = 0.5
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= SY_Del6D_ZX_5G) | (
                                TF_res_GC['pred_probability_7D'] >= SY_Del7D_ZX_5G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()
            elif ftype == '爱立信':
                if Tel_mode == '4G':
                    if distr in ['沈阳', '大连', '营口']:  # 内蒙4G爱立信 筛选 6D>=0.45 7D>=0.45 前7天退服天数>=1
                        try:
                            SY_Del6D_ZX_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_SY_4G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                            SY_Del7D_ZX_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_SY_4G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            SY_Del6D_ZX_4G = 0.55  # 80%
                            SY_Del7D_ZX_4G = 0.55
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= SY_Del6D_ZX_4G) | (
                                TF_res_GC['pred_probability_7D'] >= SY_Del7D_ZX_4G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()
                elif Tel_mode == '5G':
                    if distr in ['沈阳', '大连', '营口']:  # 内蒙5G爱立信 筛选 6D>=0.7 7D>=0.7 前7天退服天数>=1
                        try:
                            SY_Del6D_ZX_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_SY_5G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                            SY_Del7D_ZX_5G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_SY_5G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            SY_Del6D_ZX_5G = 0.7  # 78%
                            SY_Del7D_ZX_5G = 0.7
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= SY_Del6D_ZX_5G) | (
                                TF_res_GC['pred_probability_7D'] >= SY_Del7D_ZX_5G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()
            elif ftype == '大唐':
                if Tel_mode == '4G':
                    if distr in ['抚顺', '本溪', '丹东', '锦州', '葫芦岛']:  # 内蒙4G大唐 筛选 6D>=0.55 7D>=0.55 前7天退服天数>=1
                        try:
                            SY_Del6D_ZX_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod1_SY_4G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                            SY_Del7D_ZX_4G = mysql_dealdata(table_name='danger_ident_config',
                                                            key_value='Thresh_mod2_SY_4G_ZX', IF_multi=0)  # 0为一对多 1位一对一
                        except:
                            SY_Del6D_ZX_4G = 0.55
                            SY_Del7D_ZX_4G = 0.55
                        TF_res_GC_select = TF_res_GC[(TF_res_GC['未来三天退服概率'] >= SY_Del6D_ZX_4G) | (
                                TF_res_GC['pred_probability_7D'] >= SY_Del7D_ZX_4G)]  # |(TF_res_GC['规则置信度sum']!=0)
                        TF_res_GC_select = TF_res_GC_select[TF_res_GC_select['前七天退服天数'] >= 1]
                        TF_res_GC_2 = TF_res_GC_select.copy()
            TF_res_save = TF_res_GC_2[['基站id', '所属基站', 'date', '未来三天退服概率']]
            TF_res_save.columns = ['基站id', '网元名称', 'date', 'pred_probability']
            TF_res_save['厂家'] = ftype
            TF_res_save['地市'] = distr
            TF_res_save['pred_label'] = 1
            TF_res_save= TF_res_save[['基站id', '网元名称', 'date','pred_label', 'pred_probability', '厂家', '地市']]
            TF_res_save_sort = TF_res_save.sort_values(by='pred_probability', ascending=False)
            # TF_res_save_sort1 = TF_res_save_sort[TF_res_save_sort['预测未来三天退服概率']>=0.5]
            TF_res_save_sort1 = TF_res_save_sort.copy()
            # print(TF_res_save_sort1)
            TF_res_save_sort1['pred_probability'] = TF_res_save_sort1['pred_probability'].map(lambda x: '%.2f%%' % (x * 100))
            TF_res_save_sort1.to_csv(
                r"{}/OutService_{}_{}_old_{}.csv".format(para.out_path, Inspect_date, distr_new, Tel_mode), index=False,
                encoding='gbk')
            # ---------------- 20210623 剔除夜间节电基站 --------------------------
            jiedian = pd.read_excel(r"./Data/汇总/Project_Opt/夜间节电基站清单.xlsx")
            TF_res_save_sort1_new = pd.merge(TF_res_save_sort1, jiedian['基站名称A'], left_on='网元名称', right_on='基站名称A',
                                             how='left')
            TF_res_save_sort2 = TF_res_save_sort1_new[TF_res_save_sort1_new['基站名称A'].isnull()]
            TF_res_save_sort2.drop(['基站名称A'], axis=1, inplace=True)
            # TF_res_save_sort2.to_csv(r"{}/OutService_{}_{}_{}.csv".format(para.out_path, Inspect_date, distr_new,Tel_mode), index=False,encoding='gbk')

            mkdir(r"{}/OutService_{}".format(para.out_path, Inspect_date))
            # TF_res_save_sort2['制式'] = Tel_mode
            # TF_res_save_sort2.to_csv(r"{}/OutService_{}/OutService_{}_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date,distr_new,Tel_mode), encoding='gbk', index=False)

            Inspect_date_7 = datetime.datetime.strftime(base_date + datetime.timedelta(days=7), "%Y%m%d")
            Inspect_date_XJ = str(Inspect_date) +'-'+ str(Inspect_date_7)[4:8]
            TF_res_save_sort2.to_csv('{}/OutService_{}/{}_{}_predict_result_{}_{}.csv'.format(para.out_path, Inspect_date, distr, ftype,
                Tel_mode, Inspect_date_XJ),encoding='utf-8', index=False)
            TF_res_save_sort2.to_csv('{}/{}/{}_{}_predict_result_{}_{}.csv'.format(para.out_path_pre, Tel_mode, distr, ftype,
                Tel_mode, Inspect_date_XJ), encoding='utf-8',index=False)

            # ----------- 20220804 增加复制文件到服务器------------
            tuifuyuce = '{}/{}/{}_{}_predict_result_{}_{}.csv'.format(para.out_path_pre, Tel_mode, distr, ftype,Tel_mode, Inspect_date_XJ)
            cmd = 'scp {} scpuser@10.241.106.108:{}'.format(tuifuyuce, '/data/resources/result_predict/{}/'.format(Tel_mode))
            print(cmd)
            try:os.system(cmd)
            except:print('{}退服预测清单上传服务器失败'.format(Tel_mode))

            # ------------------------ 退服概率匹配到巡检清单 ------------------
            TF_res_GC = TF_res_GC[['所属基站', '未来三天退服概率']]
            Res = pd.merge(XJ_res, TF_res_GC, how='left')
            Res.to_csv("./Data/{}/Inspect_List/{}/Res_{}_{}.csv".format(distr, ftype, date, Tel_mode), index=False,
                       encoding='gbk')
        # # -------------- 20211018 更新数据合并方式 20220711 内蒙注释合并数据 --------------
        # TF_data_Re = pd.DataFrame()
        # IF_concat_TF = 0
        # for Tel_mode in ['4G', '5G']:
        #     try:
        #         TF_data = pd.read_csv(
        #             "{}/OutService_{}/OutService_{}_{}_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date,
        #                                                                  distr_new, Tel_mode, ftype), encoding='gbk')
        #         TF_data = TF_data.drop_duplicates(subset=['基站id'], keep='first')
        #         print('{}退服数据存在'.format(Tel_mode))
        #         TF_data_Re = pd.concat([TF_data_Re, TF_data], axis=0)  # A.append(C_data)
        #         IF_concat_TF = 1
        #     except:
        #         print('{}退服数据不存在'.format(Tel_mode))
        # if IF_concat_TF == 0:
        #     print('退服数据合并异常')
        # TF_data_Re.to_csv(
        #     "{}/OutService_{}/OutService_{}_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date, distr_new,
        #                                                       ftype), encoding='gbk', index=False)


def num2chi(num):
    '''转换[0, 10000)之间的阿拉伯数字
    '''
    _MAPPING = (u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九',)
    _P0 = (u'', u'十', u'百', u'千',)
    _MIN, _MAX = 0, 9999999999999999
    assert (0 <= num and num < 10000)
    if num < 10:
        return _MAPPING[num]
    else:
        lst = []
        while num >= 10:
            lst.append(num % 10)  # lst=[8,]
            num = int(num / 10)
            lst.append(num)
            c = len(lst)  # 位数
            result = u''
        for idx, val in enumerate(lst):
            if val != 0:
                result += _P0[idx] + _MAPPING[val]
            if idx < c - 1 and lst[idx + 1] == 0:
                result += u'零'
        return result[::-1].replace(u'一十', u'十')


def mktime_gap(yy, mm, dd, days):
    base_date = datetime.date(yy, mm, dd)
    if days >= 0:
        Inspect_date1 = datetime.datetime.strftime(base_date + datetime.timedelta(days=days), "%Y%m%d")
        Inspect_date2 = datetime.datetime.strftime(base_date + datetime.timedelta(days=(days + 6)), "%Y%m%d")
        Inspect_date = Inspect_date1 + '-' + Inspect_date2  # 巡检的日期 储存文件名
    else:
        days = abs(days)
        Inspect_date1 = datetime.datetime.strftime(base_date - datetime.timedelta(days=days), "%Y%m%d")
        Inspect_date2 = datetime.datetime.strftime(base_date - datetime.timedelta(days=(days - 6)), "%Y%m%d")
        Inspect_date = Inspect_date1 + '-' + Inspect_date2  # 巡检的日期 储存文件名
    return Inspect_date


def list_remark(para):
    distr_list = para.distr_list
    ftype = para.ftype
    Num_JZ = para.list_num
    date = para.date
    yy1 = int(date.split('-')[1][0:4])
    mm1 = int(date.split('-')[1][4:6])
    dd1 = int(date.split('-')[1][6:8])
    base_date = datetime.date(yy1, mm1, dd1)
    Inspect_date_one = datetime.datetime.strftime(base_date, "%Y%m%d")
    Inspect_date = datetime.datetime.strftime(base_date + datetime.timedelta(days=1), "%Y%m%d")
    Inspect_date_7 = datetime.datetime.strftime(base_date + datetime.timedelta(days=7), "%Y%m%d")
    Inspect_date1 = datetime.datetime.strftime(base_date - datetime.timedelta(days=1), "%Y%m%d")
    Inspect_date2 = datetime.datetime.strftime(base_date - datetime.timedelta(days=2), "%Y%m%d")
    Inspect_date3 = datetime.datetime.strftime(base_date - datetime.timedelta(days=3), "%Y%m%d")

    date1 = pd.date_range(date.split('-')[0], date.split('-')[1])
    date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()
    # week_0 = datetime.date(yy, mm, dd).isocalendar()[-2]
    # week_lab0, week_lab1, week_lab2, week_lab3 = map(lambda x: num2chi(x),
    #                                                  list(reversed([x for x in range(week_0 - 3, week_0 + 1)])))
    temp_gd_list = []
    for file_date in date_list:
        try:
            # temp_gd = pd.read_excel('{}/order_{}_time.xlsx'.format(para.ord_path, file_date))
            temp_gd = pd.read_excel('{}/order_{}_time.xlsx'.format(para.ord_path, file_date))
            temp_gd_list.append(temp_gd)
        except:
            print('order_{}_time.xlsx is not found'.format(file_date))

    try:
        GD_list = pd.concat(temp_gd_list, axis=0)
    except:
        GD_list = pd.DataFrame()
        print('----------------不存在历史七天工单数据----------------')

    # temp_gd_list = []
    # for file in os.listdir('./Data/汇总/Fault_Ord/'):
    #     file_date = int(file.split('_')[-1].split('.')[0])
    #     if (file_date >= low) & (file_date <= high):
    #         temp_gd = pd.read_excel('./Data/汇总/Fault_Ord/order_{}.xlsx'.format(file_date))
    #         temp_gd_list.append(temp_gd)
    # GD_list = pd.concat(temp_gd_list, axis=0)

    '''---------------- 工程预约服务器部署代码 ---------------'''
    # temp_rese_list = []
    # for file_date in date_list:
    #     try:
    #         temp_ord = pd.read_excel('./Data/汇总/Reserve_List/{}工程预约清单.xlsx'.format(file_date))
    #     except:
    #          print('{}工程预约清单.xlsx is not found'.format(file_date))
    #
    #     pattern = re.compile('[-_]\d+?')
    #     temp_ord['NENAME'] = temp_ord['NENAME'].map(lambda x:pattern.sub('', str(x)))
    #     temp_ord = temp_ord.drop_duplicates(['NENAME'])
    #     temp_ord['日期'] = pd.to_datetime(file_date, format='%Y%m%d')
    #     temp_ord['地市'] = temp_ord['REGION_NAME'].map(lambda x:str(x).replace('地区',''))
    #     temp_rese_list.append(temp_ord)
    # Pres_list_all_1 = pd.concat(temp_rese_list, axis=0)

    '''---------------- 工程预约本地调试代码 ---------------'''
    temp_rese_list = []
    temp_ord = pd.read_excel('{}/20210102工程预约清单.xlsx'.format(para.preserve_path))
    pattern = re.compile('[-_]\d+?')
    temp_ord['NENAME'] = temp_ord['NENAME'].map(lambda x: pattern.sub('', str(x)))
    temp_ord = temp_ord.drop_duplicates(['NENAME'])
    temp_ord['日期'] = pd.to_datetime(file_date, format='%Y%m%d')
    temp_ord['地市'] = temp_ord['REGION_NAME'].map(lambda x: str(x).replace('地区', ''))
    temp_rese_list.append(temp_ord)
    Pres_list_all_1 = pd.concat(temp_rese_list, axis=0)

    Pres_list_all_1.rename(columns={'NENAME': '站点'}, inplace=True)
    Pres_list_all_2 = Pres_list_all_1[['日期', '站点', '地市']]
    Pres_list_all_2['日期'] = Pres_list_all_2['日期'].astype(str)
    Pres_list_all_2['week日期'] = Pres_list_all_2.groupby('站点')['日期'].transform(lambda x: ';'.join(x))
    Pres_list_all = Pres_list_all_2.drop_duplicates(['站点'])
    Pres_list_all.rename(columns={'week日期': '预约站时间'}, inplace=True)
    Pres_list_all = Pres_list_all[['预约站时间', '站点', '地市']]

    # Pres_list_all = pd.read_excel('./Data/{}/Reserve_List/预约站点清单_20201222.xlsx'.format(distr))

    gongcanfile = "./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv"
    try:
        GongCan_o = pd.read_csv(gongcanfile, encoding='gbk', engine='python')
    except:
        GongCan_o = pd.read_csv(gongcanfile, encoding='utf-8', engine='python')
    # ---- 20220628 内蒙从工参中读保障级别 ----
    # vipfile = "./Data/汇总/Project_Opt/全网场景清单.xlsx"
    # VIP_list_o = pd.read_excel(vipfile)
    VIP_list = GongCan_o[['基站名称', 'VIP级别']]
    VIP_list.rename(columns={'基站名称': '基站', 'VIP级别': '保障场景'}, inplace=True)
    VIP_list['级别'] = 1
    # ---- End 20220628 内蒙从工参中读保障级别 ----

    GongCan = GongCan_o[['基站名称', '所属机房/位置点']]
    GongCan.drop_duplicates(subset='基站名称', inplace=True)
    GongCan.rename(columns={'基站名称': '所属基站', '所属机房/位置点': '所属机房'}, inplace=True)
    # # ---- 20220628 内蒙工单中就是基站名称 不需要匹配工参
    # GongCan_m = GongCan_o[['小区中文名', '基站名称']]  # 读取工参 筛选小区和基站的关系表
    # GongCan_m.drop_duplicates(subset='小区中文名', inplace=True)
    # GongCan_m.rename(columns={'小区中文名': '网元名称'}, inplace=True)
    # # ---- End 20220628 内蒙工单中就是基站名称 不需要匹配工参
    for distr in distr_list:
        print(distr + '巡检')
        for Tel_mode in ['4G', '5G']:
            try:
                Inspect_org_0 = pd.read_csv(
                    "./Data/{}/Inspect_List/{}/Res_{}_{}.csv".format(distr, ftype, date, Tel_mode), encoding='gbk',
                    engine='python')
            except:
                print("没有文件/Data/{}/Inspect_List/{}/Res_{}_{}.csv".format(distr, ftype, date, Tel_mode))
                continue
            Inspect_org = Inspect_org_0[~(Inspect_org_0['基站健康度异常程度'] == 0)]

            def add_weight(x):
                if (x > 0.4) & (x <= 0.5):
                    return 1.2
                elif (x > 0.5) & (x <= 0.6):
                    return 1.35
                elif (x > 0.6) & (x <= 0.7):
                    return 1.5
                elif (x > 0.7) & (x <= 0.8):
                    return 1.65
                elif (x > 0.8) & (x <= 0.9):
                    return 1.8
                elif (x > 0.9) & (x <= 1):
                    return 1.95
                else:
                    return 1

            # Inspect_org['mul_weight'] = Inspect_org['未来三天退服概率'].map(lambda x:x)
            Inspect_org['mul_weight'] = Inspect_org['未来三天退服概率'].map(lambda x: add_weight(x))
            Inspect_org['基站健康度异常程度'] = Inspect_org['基站健康度异常程度'].mul(Inspect_org['mul_weight'], axis=0)
            Inspect_org.drop(['mul_weight'], axis=1, inplace=True)
            ##增加vip权重
            Inspect_org_1 = pd.merge(Inspect_org, VIP_list, left_on='所属基站', right_on='基站', how='left')
            Inspect_org_1['保障场景'] = Inspect_org_1['保障场景'].fillna('未查询到')
            weight_map = {'VVIP': 1.4,
                          'VIP': 1.2,
                          '非VIP': 1,
                          '未查询到': 1}
            Inspect_org_1['mul_weight0'] = Inspect_org_1['保障场景'].map(lambda x: weight_map[str(x)])
            Inspect_org_1['基站健康度异常程度'] = Inspect_org_1['基站健康度异常程度'].mul(Inspect_org_1['mul_weight0'], axis=0)
            Inspect_org_1.drop(['mul_weight0', '级别'], axis=1, inplace=True)
            Inspect_org = Inspect_org_1.sort_values(by='基站健康度异常程度', ascending=False)
            Inspect_org.drop(['巡检优先级'], axis=1, inplace=True)
            Inspect_org['巡检优先级'] = [i + 1 for i in range(len(Inspect_org))]
            Inspect_org = Inspect_org[
                ['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '重点关注影响业务告警发生频次', '未来三天退服概率', '所属基站', 'enbid', '保障场景', '备注', '推断模型',
                 '告警时间戳']]
            Inspect_org.drop(['推断模型'], axis=1, inplace=True)
            Alert_list = pd.read_csv(
                './Data/{}/Alert_Deal/Samp_predict_{}/故障_处理_{}_{}.csv'.format(distr, ftype, date, Tel_mode),
                encoding='gbk')
            # gongcanfile = "./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv"
            Pres_list = Pres_list_all[Pres_list_all['地市'] == distr]
            Inspect_cot = Inspect_org.copy()
            # -------- 20210616省公司需求匹配重点关注项目所属类别 202204修改-----------
            Alert_X11 = pd.read_excel('./Data/汇总/Project_Opt/4G5G无线告警分类.xlsx')
            Alert_Class1 = Alert_X11[['告警标题', '专题大类']]

            def Deal_Class1(x):
                Alert_num = x.split(',')
                Alert_sum = []
                for Alert_i in Alert_num:
                    Alert_name = Alert_i.split('：')[0].strip(' ')
                    Alert_sum.append(Alert_name)
                Alert_pd = pd.DataFrame(np.array(Alert_sum), columns=['告警标题'])
                # Alert_pd.reset_index(inplace=True)
                Alert_meg = pd.merge(Alert_pd, Alert_Class1, on='告警标题', how='left')
                Alert_meg.drop_duplicates(subset='专题大类', inplace=True)
                Alert_fb = Alert_meg['专题大类'].fillna('其他').values.tolist()
                try:
                    Alert_fb.remove(' ')
                except:
                    a = 1
                Alert_fb_jo = (',').join(Alert_fb)
                return Alert_fb_jo

            Inspect_cot['重点关注项目所属类别'] = Inspect_cot['重点关注影响业务告警发生频次'].map(lambda x: Deal_Class1(x))

            Inspect_res = Inspect_cot[:1000]  # 对所有工单都处理
            # Inspect_res = Inspect_cot[:400]

            AddJF = pd.merge(Inspect_res, GongCan, on='所属基站', how='left')

            def week_list(isp_date0, i):
                path = r"./Data/{}/Inspect_List/{}/{}_清单_{}_{}.csv".format(distr, ftype, distr, isp_date0, Tel_mode)
                if os.path.exists(path):
                    Ole_List = pd.read_csv(
                        r"./Data/{}/Inspect_List/{}/{}_清单_{}_{}.csv".format(distr, ftype, distr, isp_date0, Tel_mode),
                        encoding='gbk', engine='python')
                    Ole_List_1 = Ole_List[['所属基站', '巡检优先级']]
                    Ole_List_1.drop_duplicates(subset='所属基站', inplace=True)
                    Ole_List_1.rename(columns={'巡检优先级': '前{}周优先级'.format(i)}, inplace=True)
                else:
                    Ole_List = {'前{}周优先级'.format(i): [''], '所属基站': ['']}
                    Ole_List_1 = pd.DataFrame(Ole_List)
                return Ole_List_1

            for i, isp_date in enumerate([Inspect_date1, Inspect_date2, Inspect_date3], start=1):
                Ole_List_1 = week_list(isp_date, i)
                AddJF = pd.merge(AddJF, Ole_List_1, on='所属基站', how='left')

            AddJF_cal = AddJF.copy()
            AddJF_cal = AddJF_cal[['前1周优先级', '前2周优先级', '前3周优先级']]
            AddJF_cal = AddJF_cal.fillna(value=0)
            AddJF_cal = AddJF_cal.applymap(lambda x: 1 if x != 0 else x)
            AddJF_cal['连续问题周数'] = AddJF_cal.apply(lambda x: x.sum() + 1, axis=1)
            Finla_res = pd.concat([AddJF, AddJF_cal['连续问题周数']], axis=1)
            Finla_res = Finla_res[
                ['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '重点关注影响业务告警发生频次', '未来三天退服概率', '所属基站', 'enbid', '保障场景', '备注', '所属机房',
                 '连续问题周数', '前1周优先级', '前2周优先级', '前3周优先级', '告警时间戳']]
            # ----------------- 删除掉最近N天没有告警的基站 ----------------
            Inspect_org = Finla_res
            Inspect_org_del = Inspect_org[
                Inspect_org['告警时间戳'].str.contains('0') | Inspect_org['告警时间戳'].str.contains('1') | Inspect_org[
                    '告警时间戳'].str.contains('2')]

            Inspect_org_del['巡检优先级'] = range(1, len(Inspect_org_del) + 1)
            Inspect_org_del.reset_index(inplace=True, drop=True)
            Num_JZ = 500
            Inspect_org_del = Inspect_org_del[:Num_JZ]
            ##----------追加工单信息----------
            # distr = para.distr
            date = para.date
            XJ_list = Inspect_org_del

            XJ_Ord = pd.merge(XJ_list, Pres_list[['站点', '预约站时间']], how='left', left_on='所属基站', right_on='站点')
            XJ_Ord.drop(['站点'], axis=1, inplace=True)
            XJ_Ord['预约站时间'].fillna('非工程预约站', inplace=True)

            if len(GD_list) != 0:
                GD_list_select = GD_list[GD_list['故障地市'] == distr]
                GD_list_x = GD_list_select[
                    ['工单流水号', '维护单位', '故障发生时间', '故障消除时间', '告警消除时间', '网元名称', '处理措施', '归档操作类型', '归档操作时间']]
                # # ---- 20220628 内蒙工单中就是基站名称 不需要匹配工参
                # GD_list_x1 = pd.merge(GD_list_x, GongCan_m, on='网元名称', how='left')
                #
                # GD_list_x1['基站名称'].fillna('-1', inplace=True)
                # GD_list_x1['基站名称'] = GD_list_x1.apply(lambda x: x['网元名称'] if x['基站名称'] == '-1' else x['基站名称'], axis=1)
                # GD_list_x1.drop(['网元名称'], axis=1, inplace=True)
                # GD_list_x1.rename(columns={'基站名称': '网元名称'}, inplace=True)
                # # ---- End 20220628 内蒙工单中就是基站名称 不需要匹配工参
                GD_list_x1 = GD_list_x[
                    ['工单流水号', '维护单位', '故障发生时间', '故障消除时间', '告警消除时间', '网元名称', '处理措施', '归档操作类型', '归档操作时间']]
                GD_list_0 = GD_list_x1.sort_values(by='故障发生时间')

                def CONcat(x):
                    GD_sele = GD_list_0[GD_list_0['网元名称'] == x['所属基站']]
                    Alert_sele = Alert_list[Alert_list['网元名称'] == x['所属基站']]
                    if GD_sele.empty == True:
                        return ('否')
                    elif Alert_sele.empty == True:  # 20210621有可能基站名匹配enbid再匹配基站名就不一样了
                        return ('否')
                    else:
                        GD_sele.fillna('空空', inplace=True)
                        is_GD = GD_sele.iloc[-1, -1]
                        if (is_GD == '空空') or (is_GD == '0') or (is_GD == 0):
                            return ('是，工单处理中')
                        else:
                            # ----------------- 2021-11-18 修改归档操作时间只判断最后一条工单的数据 ---------------
                            GD_sele = GD_sele.iloc[-1,]
                            Alert_time_s = Alert_sele.iloc[-1, 0]  # 告警表中告警开始时间
                            Alert_time = datetime.datetime.strptime(str(Alert_time_s), "%Y/%m/%d %H:%M:%S")
                            try:
                                GD_sele_CZ = datetime.datetime.strptime(str(GD_sele['归档操作时间']), "%Y/%m/%d %H:%M:%S")
                            except:
                                GD_sele_CZ = datetime.datetime.strptime(str(GD_sele['归档操作时间']), "%Y-%m-%d %H:%M:%S")
                            if Alert_time > GD_sele_CZ:  # 工单表中告警消除时间
                                return ('是，工单已完结({}),但仍有告警'.format(GD_sele_CZ))
                            elif Alert_time <= GD_sele_CZ:
                                return ('是，工单已完结({}),且后续无告警'.format(GD_sele_CZ))
                            # Alert_time_s = Alert_sele.iloc[-1,0] # 告警表中告警开始时间
                            # Alert_time = datetime.datetime.strptime(str(Alert_time_s), "%Y/%m/%d %H:%M:%S")
                            # GD_sele['归档操作时间'] = GD_sele['归档操作时间'].map(lambda x:datetime.datetime.strptime(str(x),"%Y/%m/%d %H:%M:%S"))
                            # if Alert_time > GD_sele.iloc[-1,-1]:  # 工单表中告警消除时间
                            #     return ('是，工单已完结({}),但仍有告警'.format(GD_sele.iloc[-1, -1]))
                            # elif Alert_time <= GD_sele.iloc[-1,-1]:
                            #     return ('是，工单已完结({}),且后续无告警'.format(GD_sele.iloc[-1,-1]))

                XJ_Ord['上周是否已派单'] = XJ_Ord.apply(CONcat, axis=1)

                def GDcat(x):
                    GD_sele = GD_list_0[GD_list_0['网元名称'] == x['所属基站']]
                    GD_sele.fillna('工单仍在处理中', inplace=True)
                    if GD_sele.empty == True:
                        return ('无')
                    else:
                        GD_sele['汇总'] = GD_sele.apply(
                            lambda x: str(x['工单流水号']) + '_' + str(x['维护单位']) + '_' + str(x['故障发生时间']) + '_' + str(
                                x['处理措施']), axis=1)
                        GD_contact = GD_sele.groupby(['网元名称'])['汇总'].apply(list).to_frame()
                        GD_contact_o = GD_contact.iloc[0, 0]
                        # GD_final = '[' + ']|\n['.join(GD_contact_o) + ']'
                        GD_contact_o = [x.strip(' ').replace('\n', '') for x in GD_contact_o]  # 改为只用'|'连接
                        GD_final = '[' + ']|['.join(GD_contact_o) + ']'
                        return GD_final

                XJ_Ord['历史工单记录'] = XJ_Ord.apply(GDcat, axis=1)

                XJ_Ord['工单完结时间'] = XJ_Ord['上周是否已派单'].map(
                    lambda x: str((re.findall(re.compile(r'[(](.*?)[)]', re.S), x))[0]) if '工单已完结' in x else '*')
                Alert_list['告警开始时间'] = pd.to_datetime(Alert_list['告警开始时间'])
                Alert_list.sort_values(by=['告警开始时间'], inplace=True)

                def After_GD(x, data_alarm):
                    BT_name = x['所属基站']
                    data_alarm = data_alarm[data_alarm['网元名称'] == BT_name]
                    if x['工单完结时间'] == '*':
                        return '无'
                    else:
                        x['工单完结时间'] = pd.to_datetime(x['工单完结时间'], format="%Y/%m/%d %H:%M:%S")
                        data_alarm = data_alarm[data_alarm['告警开始时间'] >= x['工单完结时间']]
                        alarm_pre = pd.DataFrame(data_alarm['告警名称'].value_counts())
                        alarm_pre.reset_index(drop=False, inplace=True)
                        alarm_pre.columns = ['告警名称', '发生次数']
                        IF_alarmalso_list = []
                        for n in range(len(alarm_pre)):
                            IF_alarmalso_list.append(alarm_pre.iloc[n, 0] + '：' + str(alarm_pre.iloc[n, 1]) + '次')
                        IF_alarmalso = '，'.join(IF_alarmalso_list)
                        return IF_alarmalso

                XJ_Ord['工单处理后仍发生告警频次'] = XJ_Ord.apply(lambda x: After_GD(x, Alert_list), axis=1)
                XJ_Ord.drop(['工单完结时间'], axis=1, inplace=True)
            else:
                print('无历史7天工单数据')
                XJ_Ord['上周是否已派单'] = ''
                XJ_Ord['历史工单记录'] = ''
                XJ_Ord['工单处理后仍发生告警频次'] = ''
                # XJ_Ord.to_csv(r"./Data/{}/Inspect_List/{}/{}_测试_{}_old_{}.csv".format(distr, ftype, distr, Inspect_date,Tel_mode))
            # ----------------- 20210611对处理完工单仍有告警'但仍有告警'的加权重 -------------
            XJ_Ord['基站健康度异常程度'] = XJ_Ord.apply(
                lambda x: x['基站健康度异常程度'] * 1.5 if '但仍有告警' in x['上周是否已派单'] else x['基站健康度异常程度'], axis=1)
            XJ_Ord = XJ_Ord.sort_values(by='基站健康度异常程度', ascending=False)

            ##---------------保留异常程度超过4%的告警-------------- # 20210526 保留异常程度大于0的告警 避免告警种类过少
            def Deal_retain(x):
                Alert_num = x.split(',')
                Alert_sum = []
                for Alert_i in Alert_num:
                    Alert_name = re.split('[()]', Alert_i)[-3]
                    Alert_pro = float(re.split('[()]', Alert_i)[-2].strip("%")) / 100
                    if Alert_pro >= 0:
                        Alert_sum.append(Alert_name)
                Alert_fb_jo = (',').join(Alert_sum)
                return Alert_fb_jo

            XJ_Ord['关注项目(保留)'] = XJ_Ord['巡检重点关注项目'].map(lambda x: Deal_retain(x))

            # # ---------------写重复了？20220406注释 增加重点关注项目所属类别-------------
            # Alert_X1 = pd.read_excel('./Data/汇总/Project_Opt/4G5G无线告警分类.xlsx')
            # Alert_Class = Alert_X1[['告警标题', '专题大类']]
            # def Deal_Class(x):
            #     Alert_num = x.split(',')
            #     Alert_sum = []
            #     for Alert_i in Alert_num:
            #         Alert_name = Alert_i.split('：')[0].strip(' ')
            #         Alert_sum.append(Alert_name)
            #     Alert_pd = pd.DataFrame(np.array(Alert_sum), columns=['告警标题'])
            #     # Alert_pd.reset_index(inplace=True)
            #     Alert_meg = pd.merge(Alert_pd, Alert_Class, on='告警标题', how='left')
            #     Alert_meg.drop_duplicates(subset='专题大类', inplace=True)
            #     Alert_fb = Alert_meg['专题大类'].fillna('-1').values.tolist()
            #     try:
            #         Alert_fb.remove(' ')
            #     except:
            #         a = 1
            #     Alert_fb_jo = (',').join(Alert_fb)
            #     return Alert_fb_jo
            #
            XJ_Ord['重点关注项目所属类别'] = XJ_Ord['重点关注影响业务告警发生频次'].map(lambda x: Deal_Class1(x))

            # ---------------增加告警处理建议 202204修改--------------
            Alert_X2 = pd.read_excel('./Data/汇总/Project_Opt/4G5G无线告警梳理建议.xlsx')
            Alert_Way = Alert_X2[['告警标题', '处理方案建议']]

            def Deal_Way(x):
                Alert_num = x.split(',')
                Alert_sum = []
                for Alert_i in Alert_num:
                    Alert_name = Alert_i.split('：')[0].strip(' ')
                    Alert_sum.append(Alert_name)
                Alert_pd = pd.DataFrame(np.array(Alert_sum), columns=['告警标题'])
                # Alert_pd.reset_index(inplace=True)
                Alert_meg = pd.merge(Alert_pd, Alert_Way, on='告警标题', how='left')
                Alert_meg.drop_duplicates(subset='处理方案建议', inplace=True)
                Alert_fb = Alert_meg['处理方案建议'].fillna('-1').values.tolist()
                try:
                    Alert_fb.remove('-1')
                except:
                    a = 1
                Alert_fb_jo = ('\n').join(Alert_fb)
                return Alert_fb_jo

            XJ_Ord['告警处理措施'] = XJ_Ord['关注项目(保留)'].map(lambda x: Deal_Way(x))

            # 删除特殊重点关注项目所属类别
            XJ_Ord_1 = XJ_Ord[~((XJ_Ord['重点关注项目所属类别'].str.contains('动力')
                                 & XJ_Ord['重点关注项目所属类别'].str.contains('退服'))
                                | (XJ_Ord['重点关注项目所属类别'] == '动力')
                                | (XJ_Ord['重点关注项目所属类别'] == '退服'))]

            # 删除特殊重点关注影响业务告警
            pattern = re.compile('[(]+?.+?[)]+?')

            def spl_sort(x):
                y = pattern.sub('', str(x)).replace("'", '')
                y1 = y.split(',')
                y2 = [i.strip(' ') for i in y1]
                y2.sort(reverse=True)
                return ','.join(y2)

            XJ_Ord_1['巡检重点关注项目0'] = XJ_Ord_1['巡检重点关注项目'].map(lambda x: spl_sort(x))
            Alert_X2 = pd.read_csv('./Data/汇总/Project_Opt/剔除巡检类型.csv', encoding='gbk')

            def li_sort(x):
                yl = x.split('，')
                yl.sort(reverse=True)
                return ','.join(yl)

            Alert_X2['巡检类型0'] = Alert_X2['巡检类型'].map(lambda x: li_sort(x))

            XJ_Ord_merge0 = pd.merge(XJ_Ord_1, Alert_X2['巡检类型0'], left_on='巡检重点关注项目0', right_on='巡检类型0', how='left')
            XJ_Ord_merge = XJ_Ord_merge0[XJ_Ord_merge0['巡检类型0'].isnull()]
            XJ_Ord_merge.drop(['巡检类型0', '巡检重点关注项目0'], axis=1, inplace=True)
            # 选择输出的Top清单数量
            if distr == '西安':
                XJ_Ord = XJ_Ord[:500]
            elif distr == '铜川' or distr == '商洛':
                XJ_Ord = XJ_Ord[:300]
            else:
                XJ_Ord = XJ_Ord[:400]
            XJ_Ord_merge.drop(['巡检优先级'], axis=1, inplace=True)
            XJ_Ord_merge['巡检优先级'] = [i + 1 for i in range(len(XJ_Ord_merge))]
            XJ_Ord = XJ_Ord[
                ['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '重点关注影响业务告警发生频次', '重点关注项目所属类别', '告警处理措施', '未来三天退服概率', '所属基站',
                 'enbid', '保障场景', '备注', '所属机房',
                 '预约站时间', '上周是否已派单', '历史工单记录', '工单处理后仍发生告警频次', '连续问题周数', '前1周优先级', '前2周优先级', '前3周优先级', '告警时间戳',
                 '关注项目(保留)']]
            XJ_Ord_merge = XJ_Ord_merge[
                ['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '重点关注影响业务告警发生频次', '重点关注项目所属类别', '告警处理措施', '未来三天退服概率', '所属基站',
                 'enbid', '保障场景', '备注', '所属机房',
                 '预约站时间', '上周是否已派单', '历史工单记录', '工单处理后仍发生告警频次', '连续问题周数', '前1周优先级', '前2周优先级', '前3周优先级', '告警时间戳',
                 '关注项目(保留)']]

            XJ_Ord_2 = XJ_Ord_merge[~(XJ_Ord_merge['重点关注项目所属类别'].str.contains('退服'))]
            XJ_Ord_2['巡检优先级'] = [i + 1 for i in range(len(XJ_Ord_2))]

            save_path = r"./Data/{}/Inspect_List/{}/{}_清单_{}_all_{}.csv".format(distr, ftype, distr, Inspect_date,Tel_mode)
            save_path_1 = r"./Data/{}/Inspect_List/{}/{}_清单_{}_{}.csv".format(distr, ftype, distr, Inspect_date,
                                                                              Tel_mode)
            save_path_2 = r"./Data/{}/Inspect_List/{}/{}_清单_{}_new_{}.csv".format(distr, ftype, distr, Inspect_date,
                                                                                  Tel_mode)
            save_path_3 = r"./Data/{}/Inspect_List/{}/{}_清单_{}_sum_{}.csv".format(distr, ftype, distr, Inspect_date,
                                                                                  Tel_mode)
            XJ_Ord.to_csv(save_path, encoding='gbk', index=False) # 没有剔除巡检类型的
            XJ_Ord_merge.to_csv(save_path_1, encoding='gbk', index=False) # 剔除了动力+退服 以及所有自定义告警组合的
            XJ_Ord_2.to_csv(save_path_2, encoding='gbk', index=False) # XJ_Ord_merge基础上剔除了包含退服的
            Inspect_cot.to_csv(save_path_3, encoding='gbk', index=False) # 全部没有截断的巡检清单
            # save_path1 = r"{}/{}_清单_{}_all_{}.csv".format(para.XJ_result, distr, Inspect_date,Tel_mode)
            save_path1_1 = r"{}/{}_清单_{}_{}_{}.csv".format(para.XJ_result, distr, Inspect_date, Tel_mode, ftype)
            save_path1_2 = r"{}/{}_清单_{}_new_{}_{}.csv".format(para.XJ_result, distr, Inspect_date, Tel_mode, ftype)
            if os.path.exists(para.XJ_result):
                # XJ_Ord.to_csv(save_path1, encoding='gbk', index=False)
                XJ_Ord_merge.to_csv(save_path1_1, encoding='gbk', index=False)
                XJ_Ord_2.to_csv(save_path1_2, encoding='gbk', index=False)
                print("清单生成完成，结果保存在: {}。".format(save_path1_1))
            else:
                print('ERROR:{} is not exists!'.format(para.XJ_result))

            # ------------------------- 输出到服务器的巡检清单 ------------------------
            XJ_Ord_3_V1 = XJ_Ord_merge[~XJ_Ord_merge['上周是否已派单'].str.contains('且后续无告警')]

            # ------------------------- 计算告警发生总频次 ------------------------
            def Cal_alarmnum(x):
                Alert_num = x.split(',')
                Alert_sum = 0
                for Alert_i in Alert_num:
                    Alert_name = Alert_i.split('：')[-1].replace('次', '')
                    Alert_sum += int(Alert_name)
                return Alert_sum

            XJ_Ord_3_V1['告警统计'] = XJ_Ord_3_V1['重点关注影响业务告警发生频次'].map(lambda x: Cal_alarmnum(x))
            if distr == '西安':
                XJ_Ord_3 = XJ_Ord_3_V1[XJ_Ord_3_V1['告警统计'] >= 5]
            elif distr == '铜川' or distr == '商洛':
                XJ_Ord_3 = XJ_Ord_3_V1[XJ_Ord_3_V1['告警统计'] > 1]
            else:
                XJ_Ord_3 = XJ_Ord_3_V1[XJ_Ord_3_V1['告警统计'] >= 1]
            # XJ_Ord_3 = XJ_Ord_3_V1[~(XJ_Ord_3_V1['重点关注项目所属类别'] == '退服')]
            XJ_Ord_3['巡检优先级'] = [i + 1 for i in range(len(XJ_Ord_3))]
            XJ_pre_rat = 1
            # 最后输出的Top清单数量
            if distr == '西安':
                XJ_Ord_3 = XJ_Ord_3.head(int(50 * XJ_pre_rat))
            elif distr == '铜川' or distr == '商洛':
                XJ_Ord_3 = XJ_Ord_3.head(int(20 * XJ_pre_rat))
            else:
                XJ_Ord_3 = XJ_Ord_3.head(int(100 * XJ_pre_rat))

            save_path1_3 = r"{}/{}_清单_{}_{}_{}.csv".format(para.out_path, distr, Inspect_date, Tel_mode, ftype)
            if len(XJ_Ord_3) == 0:
                print('{}巡检清单长度为0'.format(distr))
                continue
            XJ_Ord_3.to_csv(save_path1_3, encoding='gbk', index=False)

            # ------------------------- 2021-11-18 辽宁更新输出到web系统的巡检清单 ------------------------
            GongCan_Mul = GongCan_o[['基站名称', '覆盖类型', '覆盖场景', '所属地市', '所属区县', '设备厂家', '生命周期状态', '所属业务系统']]
            GongCan_Mul.drop_duplicates(subset='基站名称', inplace=True)
            GongCan_Mul.rename(columns={'基站名称': '所属基站'}, inplace=True)

            List_Before = XJ_Ord_3.copy()
            XJ_resu_lt = pd.merge(List_Before, GongCan_Mul, on='所属基站', how='left')

            Anl_low = datetime.datetime.strftime(base_date - datetime.timedelta(days=6), "%Y-%m-%d")
            Anl_high = datetime.datetime.strftime(base_date - datetime.timedelta(days=0), "%Y-%m-%d")
            XJ_resu_lt['分析周期开始时间'] = Anl_low
            XJ_resu_lt['分析周期结束时间'] = Anl_high
            XJ_resu_lt['历史工单中故障原因类别'] = '空'
            XJ_resu_lt['站号'] = XJ_resu_lt['enbid']
            XJ_resu_lt['所属区县分公司'] = XJ_resu_lt['所属地市']
            XJ_result_X1 = XJ_resu_lt[['分析周期开始时间', '分析周期结束时间', '巡检优先级', '基站健康度异常程度', '巡检重点关注项目',
                                       '重点关注影响业务告警发生频次', '重点关注项目所属类别', '未来三天退服概率', '所属基站', '备注',
                                       '连续问题周数', '前1周优先级', '前2周优先级', '前3周优先级', '告警时间戳', '上周是否已派单',
                                       '工单处理后仍发生告警频次', '历史工单记录', '历史工单中故障原因类别', 'enbid', '站号', '所属机房',
                                       '保障场景', '生命周期状态', '设备厂家', '所属地市', '所属区县', '所属区县分公司', '覆盖类型', '所属业务系统','告警处理措施']]
            XJ_result_X1.columns = ['分析周期开始时间', '分析周期结束时间', '督办整治优先级', '基站健康度异常程度', '整治重点告警项目',
                                    '重点关注影响业务告警发生频次', '故障原因类别', '预测未来退服概率', '所属基站', '备注',
                                    '连续异常问题周数', '前1周优先级', '前2周优先级', '前3周优先级', '本周发生异常告警的时间戳（周几）', '是否已派单',
                                    '工单处理后仍发生告警频次', '本周已派工单记录', '历史工单中故障原因类别', 'ENODEBID', '站号', '所属机房',
                                    'VIP级别', '生命周期状态', '厂家', '所属地市', '所属区县', '所属区县分公司', '覆盖类型', '制式','告警处理措施']
            XJ_result_X1.fillna(0, inplace=True)  # 如果空的话会报错
            Inspect_date_XJ = str(Inspect_date) +'-'+ str(Inspect_date_7)[4:8]
            with open((r"{}/{}/试点第一批_{}_{}_Top异常基站清单_{}_{}.csv".format(para.out_path_web, Tel_mode,distr, ftype, Tel_mode, Inspect_date_XJ)),
                      'w', encoding='utf-8') as f:
                f.write('{}\n'.format('$$'.join(XJ_result_X1.columns)))
                for i in range(len(XJ_result_X1)):
                    b = '$$'.join([str(c) for c in XJ_result_X1.loc[i].values.tolist()])
                    b = b.replace("\n", " ")
                    f.write('{}\n'.format(b))

            # ----------- 20220804 增加复制文件到服务器------------
            yichangqingdan = "{}/{}/试点第一批_{}_{}_Top异常基站清单_{}_{}.csv".format(para.out_path_web, Tel_mode,distr, ftype, Tel_mode, Inspect_date_XJ)
            cmd = 'scp {} scpuser@10.241.106.108:{}'.format(yichangqingdan, '/data/resources/result/{}/'.format(Tel_mode))
            print(cmd)
            try:os.system(cmd)
            except:print('{}Top异常基站清单上传服务器失败'.format(Tel_mode))
            # ------------------------- End 2021-11-18 辽宁更新输出到web系统的巡检清单 ------------------------
            # ---------------------- 2021-1-12 内蒙暂时注释 ---------------------------
            # yy1 = int(date.split('-')[1][0:4])
            # mm1 = int(date.split('-')[1][4:6])
            # dd1 = int(date.split('-')[1][6:8])
            # base_date = datetime.date(yy1, mm1, dd1)
            # Inspect_date = datetime.datetime.strftime(base_date + datetime.timedelta(days=1), "%Y%m%d")
            Inspect_time = datetime.datetime.strftime(base_date + datetime.timedelta(days=1), "%Y-%m-%d %H:%M:%S")
            XJ_Ord_Out_V2 = XJ_Ord_3

            def Del_Alarm_num(x):
                Alert_num = x.split(',')
                Alert_sum = []
                for Alert_i in Alert_num:
                    Alert_name = Alert_i.split(')')[1] + ')'
                    Alert_sum.append(Alert_name)
                Alert_fb_jo = (',').join(Alert_sum)
                return Alert_fb_jo

            XJ_Ord_Out_V2['巡检重点关注项目1'] = XJ_Ord_Out_V2['巡检重点关注项目'].map(lambda x: Del_Alarm_num(x))

            def New_items(x):
                len_items = len(x['巡检重点关注项目1'].split(','))
                items_group = []
                for i in range(len_items):
                    items_group.append(
                        x['巡检重点关注项目1'].split(',')[i] + '：' + x['重点关注影响业务告警发生频次'].split(',')[i].split('：')[1])  #
                items_group_jo = (',').join(items_group)
                return items_group_jo

            XJ_Ord_Out_V2['巡检重点关注项目2'] = XJ_Ord_Out_V2.apply(lambda x: New_items(x), axis=1)
            XJ_Ord_Out_V2['告警描述'] = XJ_Ord_Out_V2.apply(
                lambda x: '巡检重点关注项目：' + x['巡检重点关注项目2'] + '；   重点关注项目所属类别：' + x['重点关注项目所属类别'] + '；   未来三天退服概率：' + str(
                    '%.2f%%' % (x['未来三天退服概率'] * 100)), axis=1)

            def XJ_list_Gen(distr, XJ_Ord_Out):
                ftype_dict_num = {'华为': '01', '中兴': '02'}
                ftype_num = ftype_dict_num.get(ftype)
                if Tel_mode == '4G':
                    distr_dict_num = {'拉萨':'10','昌都':'11','山南':'12','林芝':'13','日喀则':'14','那曲':'15','阿里':'16'}
                elif Tel_mode == '5G':
                    distr_dict_num = {'拉萨':'20','昌都':'21','山南':'22','林芝':'23','日喀则':'24','那曲':'25','阿里':'26'}
                distr_num = distr_dict_num.get(distr)
                XJ_Ord_Out['alarmId_bz'] = XJ_Ord_Out['巡检优先级'].map(
                    lambda x: str(x) if len(str(x)) == 3 else ('0' + str(x) if len(str(x)) == 2 else '00' + str(x)))
                XJ_Ord_Out['alarmId'] = Inspect_date + ftype_num + distr_num + XJ_Ord_Out['alarmId_bz']
                XJ_Ord_Out['alarmSeq'] = np.random.randint(100000000, high=999999999, size=len(XJ_Ord_Out))
                XJ_Ord_Out['alarmTitle'] = '基站AI异常检测告警'
                XJ_Ord_Out['alarmStatus'] = 1
                XJ_Ord_Out['eventTime'] = Inspect_time
                XJ_Ord_Out['locationInfo'] = ' '
                XJ_Ord_Out['neName'] = XJ_Ord_Out['所属基站']
                if Tel_mode == '4G':
                    XJ_Ord_Out['neType'] = 'ENB'
                elif Tel_mode == '5G':
                    XJ_Ord_Out['neType'] = 'GNB'
                XJ_Ord_Out['objectName'] = XJ_Ord_Out['所属基站']
                if Tel_mode == '4G':
                    XJ_Ord_Out['objectType'] = 'EnbFunction'
                elif Tel_mode == '5G':
                    XJ_Ord_Out['objectType'] = 'GnbFunction'
                XJ_Ord_Out['origSeverity'] = 1
                XJ_Ord_Out['omcIP'] = 16166
                XJ_Ord_Out['standard_alarm_id'] = '0103-777-006-10-000001'
                XJ_Ord_Out['alarmCount'] = XJ_Ord_Out['关注项目(保留)']
                XJ_Ord_Out['outservicePro'] = XJ_Ord_Out['未来三天退服概率'].map(lambda x: '%.2f%%' % (x * 100))
                XJ_Ord_Out['reportGroup'] = '否'
                XJ_Ord_Out['NetWorkType'] = '1010104011106'
                XJ_Ord_Out['AlarmDesc'] = XJ_Ord_Out['告警描述']
                XJ_Ord_Out_1 = XJ_Ord_Out[
                    ['alarmId', 'alarmSeq', 'alarmTitle', 'alarmStatus', 'eventTime', 'locationInfo', 'neName',
                     'neType',
                     'objectName', 'objectType', 'origSeverity', 'omcIP', 'standard_alarm_id', 'alarmCount',
                     'outservicePro',
                     'reportGroup', 'NetWorkType', 'AlarmDesc']]
                return XJ_Ord_Out_1

            XJ_Ord_Out = XJ_list_Gen(distr, XJ_Ord_Out_V2)

            def json_write(save_path_csv, save_path_json):
                # 指定encodeing='utf-8'中文防止乱码
                csvfile = open(save_path_csv, 'r', encoding='gbk')
                jsonfile = open(save_path_json, 'w', encoding='utf-8')
                reader = csv.DictReader(csvfile)
                # 指定ensure_ascii=False 为了不让中文显示为ascii字符码
                out = json.dumps([row for row in reader], ensure_ascii=False)
                jsonfile.write(out)
                return 0

            save_path1_4 = r"{}/{}_巡检_{}_output_{}_{}.csv".format(para.out_path, distr, Inspect_date, Tel_mode, ftype)
            save_path1_5 = r"{}/{}_巡检_{}_{}_{}.json".format(para.out_path, distr, Inspect_date, Tel_mode, ftype)
            XJ_Ord_Out.to_csv(save_path1_4, encoding='gbk', index=False)
            # json_write(save_path1_4,save_path1_5)

            XJ_Ord_Out['告警清除状态'] = 0
            XJ_Ord_Out['告警清除时间'] = 0
            XJ_Ord_Out_1 = XJ_Ord_Out.copy()
            XJ_Ord_Out_1['匹配标志'] = 1
            XJ_Ord_Out_merge = XJ_Ord_Out_1[['neName', '匹配标志']]
            save_path1_sum = r"{}/{}_巡检汇总_{}_{}.csv".format(para.out_path, distr, Tel_mode, ftype)

            if os.path.exists(save_path1_sum):
                XJ_Ord_Out_sum = pd.read_csv(save_path1_sum, encoding='gbk')
                # -------------------------- 匹配未清除的告警 -----------------------------
                XJ_Ord_weiQC = XJ_Ord_Out_sum[XJ_Ord_Out_sum['告警清除状态'] == 0]
                XJ_Ord_weiQC_merge = pd.merge(XJ_Ord_weiQC, XJ_Ord_Out_merge, left_on='neName', right_on='neName',
                                              how='left')
                XJ_Ord_weiQC_merge = XJ_Ord_weiQC_merge.fillna('是')
                XJ_Ord_weiQC_merge['告警清除状态'] = XJ_Ord_weiQC_merge['匹配标志'].map(lambda x: 1 if x == '是' else 0)
                XJ_Ord_weiQC_merge['告警清除时间'] = XJ_Ord_weiQC_merge['匹配标志'].map(lambda x: Inspect_time if x == '是' else 0)
                # ------------------------- 生成清除告警 ------------------------
                try:  # 这部分是不需要的，生成的规则是之前的规则，下面会替代这部分内容 20220316
                    XJ_Ord_weiQC_select = XJ_Ord_weiQC_merge[XJ_Ord_weiQC_merge['匹配标志'] == '是']
                    XJ_Ord_weiQC_select1 = XJ_Ord_weiQC_select[
                        ['alarmId', 'alarmSeq', 'alarmTitle', 'alarmStatus', 'eventTime']]
                    XJ_Ord_weiQC_select1['alarmSeq'] = np.random.randint(100000000, high=999999999,
                                                                         size=len(XJ_Ord_weiQC_select1))
                    XJ_Ord_weiQC_select1['alarmStatus'] = 0
                    XJ_Ord_weiQC_select1['eventTime'] = Inspect_time
                    # save_path1_7 = r"{}/{}_清除_{}_output_{}_{}.csv".format(para.out_path, distr, Inspect_date,Tel_mode,ftype)
                    # XJ_Ord_weiQC_select1.to_csv(save_path1_7,index=False,encoding='GBK')
                    # save_path1_6 = r"{}/{}_清除_{}_{}_{}.json".format(para.out_path, distr, Inspect_date,Tel_mode,ftype)
                    # json_write(save_path1_7,save_path1_6)
                    # 生成清除每周一告警的清单 # 20210611改为按前两天没发生过告警才清除
                    # Inspect_time4 = datetime.datetime.strptime(Inspect_date, "%Y%m%d") # 当前时间
                    # Inspect_time_one = Inspect_time4 - datetime.timedelta(Inspect_time4.weekday())
                    # Inspect_time_one_str = str(Inspect_time_one).split(' ')[0].replace('-', '') # 20210322
                    # XJ_Ord_weiQC_select1['alarmID_day'] = XJ_Ord_weiQC_select1['alarmId'].map(lambda x: str(x)[0:8])
                    # XJ_Ord_weiQC_select1_day = XJ_Ord_weiQC_select1[XJ_Ord_weiQC_select1['alarmID_day'] == Inspect_time_one_str]
                    # XJ_Ord_weiQC_select1_day_del = XJ_Ord_weiQC_select1_day.drop(['alarmID_day'], axis=1)
                    # -------------- 20210611按前两天没有告警产生生成清除告警 ----------
                    Inspect_time4 = datetime.datetime.strptime(Inspect_date, "%Y%m%d")  # 当前时间
                    Inspect_time_one = Inspect_time4 - datetime.timedelta(days=1)
                    Inspect_time_two = Inspect_time4 - datetime.timedelta(days=2)
                    Inspect_time_one_str = str(Inspect_time_one).split(' ')[0].replace('-', '')
                    Inspect_time_two_str = str(Inspect_time_two).split(' ')[0].replace('-', '')
                    save_path1_sum_QCday = r"{}/{}_巡检汇总_week_{}_{}.csv".format(para.out_path, distr, Tel_mode, ftype)
                    XJ_week_list = pd.read_csv(save_path1_sum_QCday, encoding='gbk')
                    XJ_week_list_act = XJ_week_list[XJ_week_list['告警清除状态'] == 0]
                    Alarm_last_list = pd.read_csv(
                        './Data/{}/Alert_Data/故障_{}_{}.csv'.format(distr, Inspect_time_one_str, Tel_mode),
                        encoding='gbk')
                    Alarm_last_list1 = pd.read_csv(
                        './Data/{}/Alert_Data/故障_{}_{}.csv'.format(distr, Inspect_time_two_str, Tel_mode),
                        encoding='gbk')
                    Alarm_last_list_concat = pd.concat([Alarm_last_list, Alarm_last_list1], axis=0)
                    IFAlarmD = pd.read_csv(r"./Data/汇总/Project_Opt/判断告警是否清除.csv", encoding='gbk',
                                           engine='python')  # 20210624只统计某些重要告警标题
                    IFAlarmD_list = IFAlarmD['重要告警标题'].values.tolist()
                    s = "|".join(IFAlarmD_list)
                    Alarm_last_list_concat = Alarm_last_list_concat[Alarm_last_list_concat['告警名称'].str.contains(s)]
                    Alarm_last_list_2 = Alarm_last_list_concat[['网元名称']]
                    XJ_week_list_act_merge = pd.merge(XJ_week_list_act, Alarm_last_list_2, left_on='neName',
                                                      right_on='网元名称', how='left')
                    XJ_week_list_act_fill = XJ_week_list_act_merge.fillna('-1')
                    XJ_week_list_QC = XJ_week_list_act_fill[XJ_week_list_act_fill['网元名称'] == '-1']
                    XJ_week_list_QC_ID = XJ_week_list_QC[['alarmId']]
                    XJ_week_list_QC_ID['alarmSeq'] = np.random.randint(100000000, high=999999999,
                                                                       size=len(XJ_week_list_QC_ID))
                    XJ_week_list_QC_ID['alarmTitle'] = "基站AI异常检测告警"
                    XJ_week_list_QC_ID['alarmStatus'] = 0
                    XJ_week_list_QC_ID['eventTime'] = Inspect_time4
                    # -----------------end-----------
                    XJ_Ord_weiQC_select1_day_del = XJ_week_list_QC_ID
                    XJ_Ord_weiQC_select1_day_del['eventTime'] = XJ_Ord_weiQC_select1_day_del['eventTime'].map(
                        lambda x: str(x))
                    save_path1_day7 = r"{}/{}_清除_{}_output_day_{}_{}.csv".format(para.out_path, distr, Inspect_date,
                                                                                 Tel_mode, ftype)
                    XJ_Ord_weiQC_select1_day_del.to_csv(save_path1_day7, index=False, encoding='GBK')
                    save_path1_day6 = r"{}/{}_清除_{}_day_{}_{}.json".format(para.out_path, distr, Inspect_date, Tel_mode,
                                                                           ftype)
                    json_write(save_path1_day7, save_path1_day6)
                    # 在"巡检汇总_week"文件中匹配清除告警
                    XJ_Ord_QC_time = XJ_Ord_weiQC_select1_day_del[['alarmId', 'eventTime']]
                    XJ_Ord_QC_time.rename(columns={'eventTime': 'eventTime0'}, inplace=True)
                    save_path1_sum_QCday = r"{}/{}_巡检汇总_week_{}_{}.csv".format(para.out_path, distr, Tel_mode, ftype)
                    if os.path.exists(save_path1_sum_QCday):
                        XJ_Ord_day_Out = pd.read_csv(save_path1_sum_QCday, encoding='gbk')
                        # -------------------------- 匹配未清除的告警 -----------------------------
                        XJ_Ord_day_QC_merge = pd.merge(XJ_Ord_day_Out, XJ_Ord_QC_time, left_on='alarmId',
                                                       right_on='alarmId', how='left')
                        XJ_Ord_day_QC = XJ_Ord_day_QC_merge.fillna('是')
                        XJ_Ord_day_QC['告警清除状态'] = XJ_Ord_day_QC.apply(lambda x: 1 if x[-1] != '是' else x[-3], axis=1)
                        XJ_Ord_day_QC['告警清除时间'] = XJ_Ord_day_QC.apply(lambda x: x[-1] if x[-1] != '是' else x[-2],
                                                                      axis=1)
                        XJ_Ord_day_QC_drop = XJ_Ord_day_QC.drop('eventTime0', axis=1)
                        XJ_Ord_day_QC_drop.to_csv(save_path1_sum_QCday, encoding='gbk', index=False)

                except:
                    save_path1_8 = r"{}/{}_清除_{}_{}_{}.json".format(para.out_path, distr, Inspect_date, Tel_mode, ftype)
                    jsonfile = open(save_path1_8, 'w', encoding='utf-8')
                # -------------------------- 保存巡检清单 ------------------------
                XJ_Ord_Out_sum3 = XJ_Ord_weiQC_merge.drop(['匹配标志'], axis=1)
                XJ_Ord_Out_sum4 = XJ_Ord_Out_sum[XJ_Ord_Out_sum['告警清除状态'] == 1].append(XJ_Ord_Out_sum3)
                XJ_Ord_Out_sum1 = XJ_Ord_Out_sum4.append(XJ_Ord_Out)
                XJ_Ord_Out_sum1['alarmId'] = XJ_Ord_Out_sum1['alarmId'].map(lambda x: str(x))
                XJ_Ord_Out_sum1.drop_duplicates(subset='alarmId', keep='first', inplace=True)
                XJ_Ord_Out_sum1.to_csv(save_path1_sum, encoding='gbk', index=False)
            else:
                XJ_Ord_Out.to_csv(save_path1_sum, encoding='gbk', index=False)

            # ------------- 如果是周一的话 生成周粒度巡检清单 -------------
            Inspect_time2 = datetime.datetime.strptime(Inspect_date, "%Y%m%d")
            Inspect_date_1 = Inspect_time2.weekday()
            if Inspect_date_1 == 0:  # 改的不对 暂时无视 ---#这部分改成直接派当天生成的工单 之前是派所有未清除的告警 在陕西不会重复派单 以防万一改了 20220316
                save_path_sum_week = r"{}/{}_巡检汇总_{}_{}.csv".format(para.out_path, distr, Tel_mode, ftype)
                XJ_Ord_Out_sum_week = pd.read_csv(save_path_sum_week, encoding='gbk')
                No_Clear = XJ_Ord_Out_sum_week[XJ_Ord_Out_sum_week['告警清除状态'] == 0]
                # No_Clear = XJ_Ord_Out
                No_Clear.drop_duplicates(subset='neName', keep='last', inplace=True)
                # ---------------- 20210625 剔除夜间节电基站 --------------------------
                jiedian = pd.read_excel(r"./Data/汇总/Project_Opt/夜间节电基站清单.xlsx")
                XJ_list_JD_new = pd.merge(No_Clear, jiedian['基站名称A'], left_on='neName', right_on='基站名称A', how='left')
                XJ_list_JD_new2 = XJ_list_JD_new[XJ_list_JD_new['基站名称A'].isnull()]
                XJ_list_JD_new2.drop(['基站名称A'], axis=1, inplace=True)

                No_Clear_Out = XJ_list_JD_new2[
                    ['alarmId', 'alarmSeq', 'alarmTitle', 'alarmStatus', 'eventTime', 'locationInfo', 'neName',
                     'neType',
                     'objectName', 'objectType', 'origSeverity', 'omcIP', 'standard_alarm_id', 'alarmCount',
                     'outservicePro',
                     'reportGroup', 'NetWorkType', 'AlarmDesc']]
                save_path_week_4 = r"{}/{}_巡检_{}_output_week_{}_{}.csv".format(para.out_path, distr, Inspect_date,
                                                                               Tel_mode, ftype)
                save_path_week_5 = r"{}/{}_巡检_{}_week_{}_{}.json".format(para.out_path, distr, Inspect_date, Tel_mode,
                                                                         ftype)
                No_Clear_Out.to_csv(save_path_week_4, encoding='gbk', index=False)
                json_write(save_path_week_4, save_path_week_5)

                No_Clear_Out['告警清除状态'] = 0
                No_Clear_Out['告警清除时间'] = 0
                XJ_Ord_Out_1 = No_Clear_Out.copy()
                XJ_Ord_Out_1['匹配标志'] = 1
                XJ_Ord_Out_merge = XJ_Ord_Out_1[['neName', '匹配标志']]
                save_path1_sum_week = r"{}/{}_巡检汇总_week_{}_{}.csv".format(para.out_path, distr, Tel_mode, ftype)
                # ----------- 20210615读取week文件并保存 -------------------
                if os.path.exists(save_path1_sum_week):
                    XJ_Ord_Out_sum4 = pd.read_csv(save_path1_sum_week, encoding='gbk')
                    XJ_Ord_Out_sum1 = XJ_Ord_Out_sum4.append(XJ_Ord_Out)
                    XJ_Ord_Out_sum1['alarmId'] = XJ_Ord_Out_sum1['alarmId'].map(lambda x: str(x))
                    XJ_Ord_Out_sum1.drop_duplicates(subset='alarmId', keep='first', inplace=True)
                    XJ_Ord_Out_sum1.to_csv(save_path1_sum_week, encoding='gbk', index=False)
                else:
                    XJ_Ord_Out.to_csv(save_path1_sum_week, encoding='gbk', index=False)
                # ----------- 20210702判断是否连续4周异常且未解决 -------------------
                try:
                    save_path1_sum_week = r"{}/{}_巡检汇总_week_{}_{}.csv".format(para.out_path, distr, Tel_mode, ftype)
                    XJ_Ord_Out_sum5 = pd.read_csv(save_path1_sum_week, encoding='gbk')
                    XJ_Ord_Out_HD = XJ_Ord_Out_sum5[XJ_Ord_Out_sum5['告警清除状态'] == 0]
                    XJ_Ord_HD_count = XJ_Ord_Out_HD['neName'].groupby(XJ_Ord_Out_HD['neName']).count()
                    XJ_Ord_HD_LX = XJ_Ord_HD_count[XJ_Ord_HD_count >= 4]
                    XJ_Ord_LX_BS = pd.DataFrame(XJ_Ord_HD_LX.index.values, columns=['neName'])
                    XJ_Ord_LX_BS_merge = pd.merge(XJ_Ord_Out_sum5, XJ_Ord_LX_BS)
                    XJ_Ord_BS_drop = XJ_Ord_LX_BS_merge.drop_duplicates(subset=['neName'], keep='last')
                    XJ_Ord_BS_drop.to_csv(
                        r"{}/{}_连续4周异常_week_{}_{}_{}.csv".format(para.out_path, distr, Inspect_date, Tel_mode, ftype),
                        index=False, encoding='gbk')
                except:
                    print('生成连续4周异常基站清单异常')
                # 每周不再单独生成清除告警 20210615
                # if os.path.exists(save_path1_sum_week):
                #     XJ_Ord_Out_sum = pd.read_csv(save_path1_sum_week, encoding='gbk')
                #     # -------------------------- 匹配未清除的告警 -----------------------------
                #     XJ_Ord_weiQC = XJ_Ord_Out_sum[XJ_Ord_Out_sum['告警清除状态'] == 0]
                #     XJ_Ord_weiQC_merge = pd.merge(XJ_Ord_weiQC, XJ_Ord_Out_merge, left_on='neName', right_on='neName',
                #                                   how='left')
                #     XJ_Ord_weiQC_merge = XJ_Ord_weiQC_merge.fillna('是')
                #     XJ_Ord_weiQC_merge['告警清除状态'] = XJ_Ord_weiQC_merge['匹配标志'].map(lambda x: 1 if x == '是' else 0)
                #     XJ_Ord_weiQC_merge['告警清除时间'] = XJ_Ord_weiQC_merge['匹配标志'].map(lambda x: Inspect_time if x == '是' else 0)
                # #     # ------------------------- 生成周粒度清除告警 ------------------------
                # #     try:
                # #         XJ_Ord_weiQC_select = XJ_Ord_weiQC_merge[XJ_Ord_weiQC_merge['匹配标志'] == '是']
                # #         XJ_Ord_weiQC_select1 = XJ_Ord_weiQC_select[
                # #             ['alarmId', 'alarmSeq', 'alarmTitle', 'alarmStatus', 'eventTime']]
                # #         XJ_Ord_weiQC_select1['alarmSeq'] = np.random.randint(100000000, high=999999999,
                # #                                                              size=len(XJ_Ord_weiQC_select1))
                # #         XJ_Ord_weiQC_select1['alarmStatus'] = 0
                # #         XJ_Ord_weiQC_select1['eventTime'] = Inspect_time
                # #         save_path1_7 = r"{}/{}_清除_{}_output_week.csv".format(para.out_path, distr, Inspect_date)
                # #         XJ_Ord_weiQC_select1.to_csv(save_path1_7, index=False, encoding='GBK')
                # #         save_path1_6 = r"{}/{}_清除_{}_week.json".format(para.out_path, distr, Inspect_date)
                # #         json_write(save_path1_7, save_path1_6)
                # #     except:
                # #         save_path1_8 = r"{}/{}_清除_{}_week.json".format(para.out_path, distr, Inspect_date)
                # #         jsonfile = open(save_path1_8, 'w', encoding='utf-8')
                # #     # -------------------------- 保存巡检清单 ------------------------
                #     XJ_Ord_Out_sum3 = XJ_Ord_weiQC_merge.drop(['匹配标志'], axis=1)
                #     XJ_Ord_Out_sum4 = XJ_Ord_Out_sum[XJ_Ord_Out_sum['告警清除状态'] == 1].append(XJ_Ord_Out_sum3)
                #     XJ_Ord_Out_sum1 = XJ_Ord_Out_sum4.append(XJ_Ord_Out)
                #     XJ_Ord_Out_sum1['alarmId'] = XJ_Ord_Out_sum1['alarmId'].map(lambda x: str(x))
                #     XJ_Ord_Out_sum1.drop_duplicates(subset='alarmId', keep='first', inplace=True)
                #     XJ_Ord_Out_sum1.to_csv(save_path1_sum_week, encoding='gbk', index=False)
                # else:
                #     XJ_Ord_Out.to_csv(save_path1_sum_week, encoding='gbk', index=False)

        # ------------------------- 修改4G/5G清单的合并方式20210918 ---------------------------
        # XJ_Json_File = []
        # path_XJ_4G = "{}/{}_巡检_{}_week_4G.json".format(para.out_path, distr, Inspect_date)
        # path_XJ_5G = "{}/{}_巡检_{}_week_5G.json".format(para.out_path, distr, Inspect_date)
        # if os.path.exists(path_XJ_4G) == True:
        #     print('4G巡检清单数据存在')
        #     XJ_Json_4G = json.load(open(r"{}/{}_巡检_{}_week_4G.json".format(para.out_path, distr, Inspect_date),'r', encoding='utf-8'))
        #     if os.path.exists(path_XJ_5G) == True:
        #         print('5G巡检清单数据存在')
        #         XJ_Json_5G = json.load(open(r"{}/{}_巡检_{}_week_5G.json".format(para.out_path, distr, Inspect_date), 'r', encoding='utf-8'))
        #         XJ_Json_File += XJ_Json_4G
        #         XJ_Json_File += XJ_Json_5G
        #         with open(r"{}/{}_巡检_{}_week.json".format(para.out_path, distr, Inspect_date), 'w',encoding='utf-8') as jsonfile:
        #             json.dump(XJ_Json_File, jsonfile, ensure_ascii=False)
        #     elif os.path.exists(path_XJ_5G) == False:
        #         print('5G巡检清单数据不存在')
        #         XJ_Json_File += XJ_Json_4G
        #         with open(r"{}/{}_巡检_{}_week.json".format(para.out_path, distr, Inspect_date), 'w',encoding='utf-8') as jsonfile:
        #             json.dump(XJ_Json_File, jsonfile, ensure_ascii=False)
        # elif os.path.exists(path_XJ_4G) == False:
        #     print('4G巡检清单数据不存在')
        #     if os.path.exists(path_XJ_5G) == True:
        #         print('5G巡检清单数据存在')
        #         XJ_Json_5G = json.load(open(r"{}/{}_巡检_{}_week_5G.json".format(para.out_path, distr, Inspect_date), 'r', encoding='utf-8'))
        #         XJ_Json_File += XJ_Json_5G
        #         with open(r"{}/{}_巡检_{}_week.json".format(para.out_path, distr, Inspect_date), 'w',encoding='utf-8') as jsonfile:
        #             json.dump(XJ_Json_File, jsonfile, ensure_ascii=False)
        #     elif os.path.exists(path_XJ_5G) == False:
        #         print('5G巡检清单数据不存在')
        #         print('巡检清单数据合并异常')

        # -------------- 20211018 更新数据合并方式 --------------
        XJ_Json_File = []
        IF_concat_XJ = 0
        for Tel_mode in ['4G', '5G']:
            try:
                XJ_data = json.load(
                    open(r"{}/{}_巡检_{}_week_{}_{}.json".format(para.out_path, distr, Inspect_date, Tel_mode, ftype),
                         'r', encoding='utf-8'))
                print('{}巡检清单数据存在'.format(Tel_mode))
                XJ_Json_File += XJ_data
                IF_concat_XJ = 1
            except:
                print('{}巡检清单数据不存在'.format(Tel_mode))
        if IF_concat_XJ == 0:
            print('巡检清单数据合并异常')
        with open(r"{}/{}_巡检_{}_week_{}.json".format(para.out_path, distr, Inspect_date, ftype), 'w',
                  encoding='utf-8') as jsonfile:
            json.dump(XJ_Json_File, jsonfile, ensure_ascii=False)

        # Json_File = []
        # path_QX_4G = "{}/{}_清除_{}_day_4G.json".format(para.out_path, distr, Inspect_date)
        # path_QX_5G = "{}/{}_清除_{}_day_5G.json".format(para.out_path, distr, Inspect_date)
        # if os.path.exists(path_QX_4G) == True:
        #     print('4G清除告警数据存在')
        #     QX_Json_4G = json.load(open(r"{}/{}_清除_{}_day_4G.json".format(para.out_path, distr, Inspect_date),'r', encoding='utf-8'))
        #     if os.path.exists(path_QX_5G) == True:
        #         print('5G清除告警数据存在')
        #         QX_Json_5G = json.load(open(r"{}/{}_清除_{}_day_5G.json".format(para.out_path, distr, Inspect_date), 'r', encoding='utf-8'))
        #         Json_File += QX_Json_4G
        #         Json_File += QX_Json_5G
        #         with open(r"{}/{}_清除_{}_day.json".format(para.out_path, distr, Inspect_date), 'w',encoding='utf-8') as jsonfile:
        #             json.dump(Json_File, jsonfile, ensure_ascii=False)
        #     elif os.path.exists(path_QX_5G) == False:
        #         print('5G清除告警数据不存在')
        #         Json_File += QX_Json_4G
        #         with open(r"{}/{}_清除_{}_day.json".format(para.out_path, distr, Inspect_date), 'w',encoding='utf-8') as jsonfile:
        #             json.dump(Json_File, jsonfile, ensure_ascii=False)
        # elif os.path.exists(path_QX_4G) == False:
        #     print('4G清除告警数据不存在')
        #     if os.path.exists(path_QX_5G) == True:
        #         print('5G清除告警数据存在')
        #         QX_Json_5G = json.load(open(r"{}/{}_清除_{}_day_5G.json".format(para.out_path, distr, Inspect_date), 'r', encoding='utf-8'))
        #         Json_File += QX_Json_5G
        #         with open(r"{}/{}_清除_{}_day.json".format(para.out_path, distr, Inspect_date), 'w',encoding='utf-8') as jsonfile:
        #             json.dump(Json_File, jsonfile, ensure_ascii=False)
        #     elif os.path.exists(path_QX_5G) == False:
        #         print('5G清除告警数据不存在')
        #         print('清除告警数据合并异常')

        # -------------- 20211018 更新数据合并方式 --------------
        QX_Json_File = []
        IF_concat_QX = 0
        for Tel_mode in ['4G', '5G']:
            try:
                QX_data = json.load(
                    open(r"{}/{}_清除_{}_day_{}_{}.json".format(para.out_path, distr, Inspect_date, Tel_mode, ftype), 'r',
                         encoding='utf-8'))
                print('{}清除告警数据存在'.format(Tel_mode))
                QX_Json_File += QX_data
                IF_concat_QX = 1
            except:
                print('{}清除告警数据不存在'.format(Tel_mode))
        if IF_concat_QX == 0:
            print('清除告警数据合并异常')
        with open(r"{}/{}_清除_{}_day_{}.json".format(para.out_path, distr, Inspect_date, ftype), 'w',
                  encoding='utf-8') as jsonfile:
            json.dump(QX_Json_File, jsonfile, ensure_ascii=False)


# ---------------------- 2021-1-12 辽宁暂时注释 ---------------------------'''

def validation(para):
    distr_list = para.distr_list
    ftype = para.ftype
    date = para.date
    yy1 = int(date.split('-')[1][0:4])
    mm1 = int(date.split('-')[1][4:6])
    dd1 = int(date.split('-')[1][6:8])
    base_date = datetime.date(yy1, mm1, dd1)
    Inspect_date1 = datetime.datetime.strftime(base_date + datetime.timedelta(days=1), "%Y%m%d")
    Inspect_date_7 = datetime.datetime.strftime(base_date - datetime.timedelta(days=0), "%Y%m%d") # 20220711文件命名
    Inspect_date = datetime.datetime.strftime(base_date - datetime.timedelta(days=6), "%Y%m%d") # 20220711读前6天的预测结果
    predict_time_0 = datetime.datetime.strptime(Inspect_date1, "%Y%m%d")
    date_low_Time = predict_time_0 - datetime.timedelta(days=7)
    date_high_Time = predict_time_0 - datetime.timedelta(days=5)
    date = pd.date_range(str(date_low_Time).split(' ')[0].replace('-', ''),
                         str(date_high_Time).split(' ')[0].replace('-', ''))
    date_list = date.astype(str).map(lambda x: x.replace('-', '')).tolist()
    # 新建文件夹
    mkdir(r"{}/OutService_{}".format(para.out_path, Inspect_date1))

    for Tel_mode in ['4G', '5G']:
        for distr in distr_list:
            print(distr + '退服验证')
            distr_dict = {'拉萨':'LS','昌都':'CD','山南':'SN','林芝':'LZ','日喀则':'RKZ','那曲':'NQ','阿里':'AL'}
            distr_new = distr_dict.get(distr)
            Inspect_date_XJ = str(Inspect_date) +'-'+ str(Inspect_date_7)[4:8]
            try:
                TF_list = pd.read_csv('{}/{}/{}_{}_predict_result_{}_{}.csv'.format(para.out_path_pre, Tel_mode, distr, ftype, Tel_mode, Inspect_date_XJ),encoding='utf-8')
            except:
                print('{}无{}{}退服预测数据'.format(distr, Inspect_date_XJ,Tel_mode))
                continue
            TF_list1 = TF_list.copy()
            # TF_list1 = TF_list[TF_list['pred_probability']>=0.6]
            # TF_list1 = TF_list[TF_list['pred_label']==1]
            tempall = []
            for date_1 in date_list:
                alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(distr, date_1), encoding='gbk')
                tempall.append(alert_data_part)

            Alert_all = pd.concat(tempall, axis=0)
            # Alert_select = Alert_all[(Alert_all['告警名称'] == '射频单元维护链路异常告警')
            #                          | (Alert_all['告警名称'] == '小区不可用告警')
            #                          | (Alert_all['告警名称'] == '网元连接中断') | (Alert_all['告警名称'] == 'eNodeB退服告警') | (
            #                                      Alert_all['告警名称'] == '传输光接口异常告警')]
            # --------- 20211021 修改匹配退服告警清单 -----------
            TF_list = real_warns_5G + real_warns_4G
            TF_list_Pd = pd.DataFrame(TF_list, columns=['告警名称']).drop_duplicates()  # 剔除+去重
            Alert_select = pd.merge(Alert_all, TF_list_Pd)

            TF_count = Alert_select.groupby('基站id',as_index = False)['告警名称'].count()
            TF_sum = len(TF_count)
            # TF_count_D = TF_count.to_frame()
            # TF_count_D.reset_index(inplace=True)

            TF_count['基站id'] = TF_count['基站id'].astype(int)
            TF_merge = pd.merge(TF_list1, TF_count, on='基站id', how='left')

            TF_merge['告警名称'].fillna(0, inplace=True)
            # TF_num = len(TF_merge)
            # TF_merge_FN = TF_merge.fillna(-1)
            # TF_1_count = TF_merge_FN['告警名称'].value_counts()
            # TF_1_count_D = TF_1_count.to_frame()
            # try:
            #     False_num = TF_1_count_D.loc[-1, '告警名称']
            # except:
            #     False_num = 0
            # precision = 1 - (False_num / TF_num)

            # --------- 20220711 新增全国统一版本后评估 -----------
            alarm_data1 = Alert_all[Alert_all['告警名称'].isin(set(TF_list))]
            alarm_data1['基站id'] = alarm_data1['基站id'].astype(int)
            subdata = TF_merge.copy()
            subdata['预测日期后一周内，实际发生退服的情况'] = subdata.apply(lambda x: (''), axis=1)
            df1 = pd.DataFrame()
            i = 0
            for row in subdata.itertuples():
                wangy = getattr(row, '基站id')
                # print(wangy)
                alarm_data2 = alarm_data1[alarm_data1['基站id'] == wangy]
                # print(alarm_data2)
                case = alarm_data2['告警名称'].value_counts().sort_index(ascending=True).to_dict()
                subdata['预测日期后一周内，实际发生退服的情况'][i] = case
                df1 = pd.concat([df1, alarm_data2], ignore_index=True)
                i = i + 1
            # --------- End 20220711 新增全国统一版本后评估 -----------
            TF_merge = subdata.copy()
            TF_merge.rename(columns={'告警名称': '实际退服告警数量'}, inplace=True)
            TF_merge['预测是否准确'] = TF_merge['实际退服告警数量'].map(lambda x:'是' if x>0 else '否')

            TF_merge.columns=['基站id', '网元名称', 'date','pred_label', 'pred_probability', '厂家', '地市', '实际退服告警数量','预测日期后一周内，实际发生退服的情况', '预测是否准确']
            TF_merge =TF_merge[['基站id', '网元名称', 'date','pred_label', 'pred_probability', '厂家', '地市','预测日期后一周内，实际发生退服的情况', '预测是否准确']]
            # TF_merge.to_csv('{}/OutServiceTest_{}_{}.csv'.format(para.out_path, Inspect_date, distr_new), encoding='gbk',index=False)
            TF_merge.to_csv(
                '{}/{}/{}_{}_predict_result_{}_后评估_{}.csv'.format(para.out_path_pre, Tel_mode, distr, ftype, Tel_mode, Inspect_date_XJ),
                encoding='utf-8', index=False)
            print("退服验证清单生成完成，结果保存在: {}。".format(
                '{}/{}/{}_{}_predict_result_{}_后评估_{}.csv'.format(para.out_path_pre, Tel_mode, distr, ftype, Tel_mode, Inspect_date_XJ)))

            # ----------- 20220804 增加复制文件到服务器------------
            tuifuyanzheng = '{}/{}/{}_{}_predict_result_{}_后评估_{}.csv'.format(para.out_path_pre, Tel_mode, distr, ftype, Tel_mode, Inspect_date_XJ)
            cmd = 'scp {} scpuser@10.241.106.108:{}'.format(tuifuyanzheng, '/data/resources/result_predict/{}/'.format(Tel_mode))
            print(cmd)
            try:os.system(cmd)
            except:print('{}退服验证清单上传服务器失败'.format(Tel_mode))

# # -------------------------- 20210924 4G/5G分别验证 --------------------------
#         for Tel_mode in ['4G', '5G']:
#     # -------------------输出到服务器退服清单的匹配 ---------------------
#             distr_dict = {'西安': 'XA', '铜川': 'TC', '宝鸡': 'BJ', '咸阳': 'XY', '渭南': 'WN', '汉中': 'HZ', '安康': 'AK', '商洛': 'SL',
#                       '延安': 'YA', '榆林': 'YL'}
#             distr_new = distr_dict.get(distr)
#             try:
#                 TF_list = pd.read_csv('{}/OutService_{}_{}_{}.csv'.format(para.out_path, Inspect_date, distr_new,Tel_mode), encoding='gbk')
#             except:
#                 print('{}无{}退服预测数据'.format(distr,Inspect_date))
#                 continue
#             TF_list1 = TF_list.copy()
#             # TF_list1 = TF_list[TF_list['pred_probability']>=0.6]
#             # TF_list1 = TF_list[TF_list['pred_label']==1]
#             tempall = []
#             for date_1 in date_list:
#                 alert_data_part = pd.read_csv("./Data/{}/Alert_Data/告警日志{}.csv".format(distr, date_1), encoding='gbk')
#                 tempall.append(alert_data_part)
#
#             Alert_all = pd.concat(tempall, axis=0)
#             if Tel_mode == '4G':
#                 Alert_select = Alert_all[(Alert_all['告警名称'] == '射频单元维护链路异常告警')
#                                      rm
#                                      | (Alert_all['告警名称'] == '网元连接中断') | (Alert_all['告警名称'] == 'eNodeB退服告警') | (
#                                                  Alert_all['告警名称'] == '传输光接口异常告警')]
#             elif Tel_mode == '5G':
#                 Alert_select = Alert_all[(Alert_all['告警名称'] == '射频单元维护链路异常告警')
#                                  | (Alert_all['告警名称'] == '小区不可用告警')| (Alert_all['告警名称'] == 'eNodeB退服告警')
#                                  | (Alert_all['告警名称'] == '网元连接中断') | (Alert_all['告警名称'] == 'gNodeB退服告警') | (
#                                          Alert_all['告警名称'] == 'NR小区不可用告警')]
#             TF_count = Alert_select.groupby('基站id')['告警名称'].count()
#             TF_sum = len(TF_count)
#             TF_count_D = TF_count.to_frame()
#             TF_count_D.reset_index(inplace=True)
#
#             TF_merge = pd.merge(TF_list1, TF_count, on='基站id', how='left')
#
#             # TF_num = len(TF_merge)
#             # TF_merge_FN = TF_merge.fillna(-1)
#             # TF_1_count = TF_merge_FN['告警名称'].value_counts()
#             # TF_1_count_D = TF_1_count.to_frame()
#             # try:
#             #     False_num = TF_1_count_D.loc[-1, '告警名称']
#             # except:
#             #     False_num = 0
#             # precision = 1 - (False_num / TF_num)
#
#             TF_merge.rename(columns={'告警名称': '实际退服告警数量'}, inplace=True)
#             TF_merge['实际退服告警数量'].fillna(0,inplace=True)
#             # TF_merge.to_csv('{}/OutServiceTest_{}_{}_{}.csv'.format(para.out_path, Inspect_date, distr_new,Tel_mode), encoding='gbk',index=False)
#             TF_merge.to_csv('{}/OutService_{}/OutServiceTest_{}_{}_{}.csv'.format(para.out_path, Inspect_date1, Inspect_date, distr_new,Tel_mode), encoding='gbk',index=False)
#             print("退服验证清单生成完成，结果保存在: {}。".format('{}/OutServiceTest_{}_{}_{}.csv'.format(para.out_path, Inspect_date, distr_new,Tel_mode)))
#
#         try:
#             TF_Test_4G = pd.read_csv('{}/OutService_{}/OutServiceTest_{}_{}_4G.csv'.format(para.out_path, Inspect_date1, Inspect_date,distr_new), encoding='gbk')
#         except:
#             print('读取4G退服验证数据异常')
#             try:
#                 TF_Test_5G = pd.read_csv('{}/OutService_{}/OutServiceTest_{}_{}_5G.csv'.format(para.out_path, Inspect_date1, Inspect_date,distr_new), encoding='gbk')
#                 TF_Test_5G.to_csv(r"{}/OutService_{}/OutService_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date, distr_new),encoding='gbk', index=False)
#                 continue
#             except:
#                 print('读取4G/5G退服验证数据异常')
#         try:
#             TF_Test_5G = pd.read_csv('{}/OutService_{}/OutServiceTest_{}_{}_5G.csv'.format(para.out_path, Inspect_date1, Inspect_date,distr_new), encoding='gbk')
#         except:
#             print('读取5G退服验证数据异常')
#             try:
#                 TF_Test_4G = pd.read_csv('{}/OutService_{}/OutServiceTest_{}_{}_4G.csv'.format(para.out_path, Inspect_date1, Inspect_date,distr_new), encoding='gbk')
#                 TF_Test_4G.to_csv(r"{}/OutService_{}/OutService_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date,distr_new), encoding='gbk', index=False)
#                 continue
#             except:
#                 print('读取5G/4G退服验证数据异常')
#         try:
#             TF_Test_Con = pd.concat([TF_Test_4G,TF_Test_5G],axis=0)
#             TF_Test_Con = TF_Test_Con.drop_duplicates(subset=['基站id'], keep='first')  # 20210917有些基站一个ID对应多个基站中文名，需要去重
#             TF_Test_Con.to_csv(r"{}/OutService_{}/OutService_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date, distr_new), encoding='gbk', index=False)
#         except:
#             print('退服验证数据合并异常')
# # -------------------------- 20210924 4G/5G分别验证 --------------------------

# # -------------- 20211018 新增厂家数据合并 20220711 内蒙注释 --------------
# def factory_contact(para):
#     distr_list = ['拉萨','昌都','山南','林芝','日喀则','那曲','阿里']
#     date = para.date
#     yy1 = int(date.split('-')[1][0:4])
#     mm1 = int(date.split('-')[1][4:6])
#     dd1 = int(date.split('-')[1][6:8])
#     base_date = datetime.date(yy1, mm1, dd1)
#     Inspect_date = datetime.datetime.strftime(base_date + datetime.timedelta(days=1), "%Y%m%d")
#     for distr in distr_list:
#         print(distr)
#         distr_dict = {'拉萨':'LS','昌都':'CD','山南':'SN','林芝':'LZ','日喀则':'RKZ','那曲':'NQ','阿里':'AL'}
#         distr_new = distr_dict.get(distr)
#
#         TF_data_Re = pd.DataFrame()
#         IF_concat_TF = 0
#         for ftype in ['华为', '中兴', '爱立信', '大唐']:
#             try:
#                 TF_data = pd.read_csv(
#                     "{}/OutService_{}/OutService_{}_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date,
#                                                                       distr_new, ftype), encoding='gbk')
#                 print('{}退服数据存在'.format(ftype))
#                 TF_data_Re = pd.concat([TF_data_Re, TF_data], axis=0)  # A.append(C_data)
#                 IF_concat_TF = 1
#             except:
#                 print('{}退服数据不存在'.format(ftype))
#         if IF_concat_TF == 0:
#             print('退服数据合并异常')
#         TF_data_Re.to_csv(
#             "{}/OutService_{}/OutService_{}_{}.csv".format(para.out_path, Inspect_date, Inspect_date, distr_new),
#             encoding='gbk', index=False)
#
#         XJ_Json_File = []
#         IF_concat_XJ = 0
#         for ftype in ['华为', '中兴', '爱立信', '大唐']:
#             try:
#                 XJ_data = json.load(
#                     open(r"{}/{}_巡检_{}_week_{}.json".format(para.out_path, distr, Inspect_date, ftype), 'r',
#                          encoding='utf-8'))
#                 print('{}巡检清单数据存在'.format(ftype))
#                 XJ_Json_File += XJ_data
#                 IF_concat_XJ = 1
#             except:
#                 print('{}巡检清单数据不存在'.format(ftype))
#         if IF_concat_XJ == 0:
#             print('巡检清单数据合并异常')
#         with open(r"{}/{}_巡检_{}_week.json".format(para.out_path, distr, Inspect_date), 'w',
#                   encoding='utf-8') as jsonfile:
#             json.dump(XJ_Json_File, jsonfile, ensure_ascii=False)
#
#         QX_Json_File = []
#         IF_concat_QX = 0
#         for ftype in ['华为', '中兴', '爱立信', '大唐']:
#             try:
#                 QX_data = json.load(
#                     open(r"{}/{}_清除_{}_day_{}.json".format(para.out_path, distr, Inspect_date, ftype), 'r',
#                          encoding='utf-8'))
#                 print('{}清除告警数据存在'.format(ftype))
#                 QX_Json_File += QX_data
#                 IF_concat_QX = 1
#             except:
#                 print('{}清除告警数据不存在'.format(ftype))
#         if IF_concat_QX == 0:
#             print('清除告警数据合并异常')
#         with open(r"{}/{}_清除_{}_day.json".format(para.out_path, distr, Inspect_date), 'w',
#                   encoding='utf-8') as jsonfile:
#             json.dump(QX_Json_File, jsonfile, ensure_ascii=False)
