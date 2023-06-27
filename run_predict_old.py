import numpy as np
import pandas as pd
import os
import re
import datetime
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from run_predict_XJ_4G import predict_XJ
from run_predict_TF import predict_TF

def predict(para):

    distr_list = para.distr_list
    ftype = para.ftype
    date = para.date

    for distr in distr_list:
        XJ_res = pd.read_csv(r"./Data/{}/Inspect_List/{}/Origin_{}.csv".format(distr, ftype, date), encoding='gbk',
                             engine='python')
        TF_res_o = pd.read_csv(r"./Data/{}/Inspect_List/{}/TFPre_{}.csv".format(distr, ftype, date),
                               encoding='gbk',
                               engine='python')
        TF_res_o = TF_res_o.rename(columns={'pred_probability': '未来三天退服概率'})
        TF_res_o['基站id'] = TF_res_o['基站id'].map(lambda x: str(x).split('.')[0])
        try:
            gongcan = pd.read_csv(r"./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv", encoding='gbk', engine='python')
        except:
            gongcan = pd.read_csv(r"./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv", encoding='utf-8', engine='python')

        GongCan_W = gongcan[['基站名称','ENODEB_ID']]
        GongCan_W.rename(columns={'ENODEB_ID': '基础小区号'},inplace=True)
        GongCan_W['基础小区号'] = GongCan_W['基础小区号'].map(lambda x: str(x).split('-')[0].split(".")[0])
        GongCan_W_del = GongCan_W.drop_duplicates(subset=['基础小区号', '基站名称'], keep='first')
        GongCan_W_del = GongCan_W_del.rename(columns={'基础小区号': '基站id', '基站名称': '所属基站'})
        TF_res = TF_res_o.copy()
        TF_res_GC = pd.merge(TF_res,GongCan_W_del,how='left')
        TF_res_GC = TF_res_GC[['所属基站','未来三天退服概率']]
        Res = pd.merge(XJ_res,TF_res_GC,how='left')
        Res.to_csv("./Data/{}/Inspect_List/{}/Res_{}.csv".format(distr, ftype, date),index=False,encoding='gbk')

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
            lst.append(num % 10) #lst=[8,]
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
def mktime_gap(yy,mm,dd,days):
    base_date = datetime.date(yy,mm,dd)
    if days>=0:
        Inspect_date1 = datetime.datetime.strftime(base_date+ datetime.timedelta(days = days), "%Y%m%d")
        Inspect_date2 = datetime.datetime.strftime(base_date+ datetime.timedelta(days = (days+6)), "%Y%m%d")
        Inspect_date = Inspect_date1+'-'+Inspect_date2 #巡检的日期 储存文件名
    else:
        days = abs(days)
        Inspect_date1 = datetime.datetime.strftime(base_date - datetime.timedelta(days=days), "%Y%m%d")
        Inspect_date2 = datetime.datetime.strftime(base_date - datetime.timedelta(days=(days - 6)),"%Y%m%d")
        Inspect_date = Inspect_date1 + '-' + Inspect_date2  # 巡检的日期 储存文件名
    return Inspect_date

def list_remark(para):
    distr_list = para.distr_list
    ftype = para.ftype
    Num_JZ = para.list_num
    date = para.date
    yy = int(date.split('-')[0][0:4])
    mm = int(date.split('-')[0][4:6])
    dd = int(date.split('-')[0][6:8])
    yy1 = int(date.split('-')[1][0:4])
    mm1 = int(date.split('-')[1][4:6])
    dd1 = int(date.split('-')[1][6:8])
    base_date = datetime.date(yy1,mm1,dd1)
    Inspect_date = datetime.datetime.strftime(base_date + datetime.timedelta(days = 1), "%Y%m%d")
    Inspect_date1 = datetime.datetime.strftime(base_date - datetime.timedelta(days = 1), "%Y%m%d")
    Inspect_date2 = datetime.datetime.strftime(base_date - datetime.timedelta(days = 2), "%Y%m%d")
    Inspect_date3 = datetime.datetime.strftime(base_date - datetime.timedelta(days = 2), "%Y%m%d")

    date1 = pd.date_range(date.split('-')[0], date.split('-')[1])
    date_list = date1.astype(str).map(lambda x: x.replace('-', '')).tolist()
    # week_0 = datetime.date(yy, mm, dd).isocalendar()[-2]
    # week_lab0, week_lab1, week_lab2, week_lab3 = map(lambda x: num2chi(x),
    #                                                  list(reversed([x for x in range(week_0 - 3, week_0 + 1)])))
    temp_gd_list = []
    for file_date in date_list:
        try:
            # temp_gd = pd.read_excel('{}/order_{}_time.xlsx'.format(para.ord_path, file_date))
            temp_gd = pd.read_excel('{}/order_{}_0.xlsx'.format(para.ord_path, file_date))
        except:
            print('order_{}_0.xlsx is not found'.format(file_date))
        temp_gd_list.append(temp_gd)
    GD_list = pd.concat(temp_gd_list, axis=0)

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
    #         temp_ord = pd.read_excel('{}/{}工程预约清单.xlsx'.format(para.preserve_path,file_date))
    #     except:
    #          print('{}工程预约清单.xlsx is not found in {}'.format(file_date,para.preserve_path))
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
    temp_ord['NENAME'] = temp_ord['NENAME'].map(lambda x:pattern.sub('', str(x)))
    temp_ord = temp_ord.drop_duplicates(['NENAME'])
    temp_ord['日期'] = pd.to_datetime(file_date, format='%Y%m%d')
    temp_ord['地市'] = temp_ord['REGION_NAME'].map(lambda x:str(x).replace('地区',''))
    temp_rese_list.append(temp_ord)
    Pres_list_all_1 = pd.concat(temp_rese_list, axis=0)

    Pres_list_all_1.rename(columns={'NENAME':'站点'},inplace=True)
    Pres_list_all_2 = Pres_list_all_1[['日期','站点','地市']]
    Pres_list_all_2['日期'] = Pres_list_all_2['日期'].astype(str)
    Pres_list_all_2['week日期'] = Pres_list_all_2.groupby('站点')['日期'].transform(lambda x: ';'.join(x))
    Pres_list_all = Pres_list_all_2.drop_duplicates(['站点'])
    Pres_list_all.rename(columns={'week日期':'预约站时间'},inplace=True)
    Pres_list_all = Pres_list_all[['预约站时间','站点','地市']]

    # Pres_list_all = pd.read_excel('./Data/{}/Reserve_List/预约站点清单_20201222.xlsx'.format(distr))

    for distr in distr_list:
        Inspect_org = pd.read_csv(
            "./Data/{}/Inspect_List/{}/Res_{}.csv".format(distr, ftype, date),encoding='gbk', engine='python')
        Alert_list = pd.read_csv('./Data/{}/Alert_Deal/Samp_predict_{}/故障_处理_{}.csv'.format(distr,ftype, date),encoding='gbk')
        gongcanfile = "./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv"
        Pres_list = Pres_list_all[Pres_list_all['地市']==distr]

        dates = para.date
        date_low = dates.split('-')[0]
        date_high = dates.split('-')[-1]
        date2 = pd.date_range(date_low, date_high)
        date_list = date2.astype(str).map(lambda x: x.replace('-', '')).tolist()

        Inspect_org_del = Inspect_org.copy()
        Inspect_score = Inspect_org_del['基站健康度异常程度'].sort_values(ascending=False)
        Inspect_score.reset_index(inplace=True, drop=True)
        Inspect_org_del.drop(['基站健康度异常程度', '推断模型'], axis=1, inplace=True)
        Inspect_cot = pd.concat([Inspect_org_del, Inspect_score], ignore_index=True, axis=1)
        Inspect_cot.columns = ['巡检优先级', '巡检重点关注项目','重点关注影响业务告警发生频次' ,'所属基站', '备注', '告警时间戳','未来三天退服概率','基站健康度异常程度']
        Inspect_cot = Inspect_cot[['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '重点关注影响业务告警发生频次','所属基站', '备注', '告警时间戳','未来三天退服概率']]
        Inspect_res = Inspect_cot[:300]

        try:
            GongCan_o = pd.read_csv(gongcanfile, encoding='gbk', engine='python')
        except:
            GongCan_o = pd.read_csv(gongcanfile, encoding='utf-8', engine='python')

        GongCan = GongCan_o[['基站名称', '所属机房/位置点']]
        GongCan.drop_duplicates(subset='基站名称', inplace=True)
        GongCan.rename(columns={'基站名称': '所属基站','所属机房/位置点':'所属机房'}, inplace=True)

        AddJF = pd.merge(Inspect_res, GongCan, on='所属基站', how='left')

        def week_list(isp_date0,i):
            path = r"./Data/{}/Inspect_List/{}/{}_清单_{}.csv".format(distr, ftype, distr, isp_date0)
            if os.path.exists(path):
                Ole_List = pd.read_csv(
                        r"./Data/{}/Inspect_List/{}/{}_清单_{}.csv".format(distr, ftype, distr, isp_date0),
                        encoding='gbk', engine='python')
                Ole_List_1 = Ole_List[['所属基站', '巡检优先级']]
                Ole_List_1.rename(columns={'巡检优先级': '前{}天优先级'.format(i)}, inplace=True)
            else:
                Ole_List = {'前{}天优先级'.format(i):[''],'所属基站':['']}
                Ole_List_1 = pd.DataFrame(Ole_List)
            return Ole_List_1

        for i,isp_date in enumerate([Inspect_date1,Inspect_date2,Inspect_date3],start=1):
            Ole_List_1 = week_list(isp_date,i)
            AddJF = pd.merge(AddJF, Ole_List_1, on='所属基站', how='left')

        AddJF_cal = AddJF.copy()
        AddJF_cal = AddJF_cal[['前1天优先级', '前2天优先级', '前3天优先级']]
        AddJF_cal = AddJF_cal.fillna(value=0)
        AddJF_cal = AddJF_cal.applymap(lambda x: 1 if x != 0 else x)
        AddJF_cal['连续问题天数'] = AddJF_cal.apply(lambda x: x.sum() + 1, axis=1)
        Finla_res = pd.concat([AddJF, AddJF_cal['连续问题天数']], axis=1)
        Finla_res = Finla_res[
            ['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '重点关注影响业务告警发生频次','所属基站', '备注', '所属机房', '连续问题天数', '前1天优先级', '前2天优先级', '前3天优先级', '告警时间戳','未来三天退服概率']]
        # 删除掉最近N天没有告警的基站
        Inspect_org = Finla_res

        Inspect_org_del = Inspect_org[Inspect_org['告警时间戳'].str.contains('0') | Inspect_org['告警时间戳'].str.contains('1')]

        Inspect_org_del['巡检优先级'] = range(1, len(Inspect_org_del) + 1)
        Inspect_org_del.reset_index(inplace=True, drop=True)
        Inspect_org_del = Inspect_org_del[:Num_JZ]
        ##------追加工单信息----
        # distr = para.distr
        date = para.date
        XJ_list = Inspect_org_del

        XJ_Ord = pd.merge(XJ_list,Pres_list[['站点','预约站时间']],how='left',left_on='所属基站',right_on='站点')
        XJ_Ord.drop(['站点'], axis=1, inplace=True)
        XJ_Ord['预约站时间'].fillna('非工程预约站', inplace=True)
        GongCan_m = GongCan_o[['小区中文名', '基站名称']]  # 读取工参 筛选小区和基站的关系表
        GongCan_m.drop_duplicates(subset='小区中文名', inplace=True)
        GongCan_m.rename(columns={'小区中文名': '网元名称'}, inplace=True)

        GD_list_select = GD_list[GD_list['故障地市']==distr]
        GD_list_x = GD_list_select[['工单流水号','维护单位','故障发生时间','故障消除时间','告警消除时间','网元名称','处理措施','归档操作类型','归档操作时间']]
        GD_list_x1 = pd.merge(GD_list_x,GongCan_m,on='网元名称',how='left')

        GD_list_x1['基站名称'].fillna('-11', inplace=True)
        GD_list_x1['基站名称'] = GD_list_x1.apply(lambda x: x['网元名称'] if x['基站名称'] == '-11' else x['基站名称'], axis=1)
        GD_list_x1.drop(['网元名称'], axis=1, inplace=True)
        GD_list_x1.rename(columns={'基站名称': '网元名称'}, inplace=True)
        GD_list_x1 = GD_list_x1[['工单流水号', '维护单位', '故障发生时间', '故障消除时间', '告警消除时间', '网元名称', '处理措施', '归档操作类型', '归档操作时间']]
        GD_list_0 = GD_list_x1.sort_values(by='故障发生时间')

        def CONcat(x):
            GD_sele = GD_list_0[GD_list_0['网元名称'] == x['所属基站']]
            Alert_sele = Alert_list[Alert_list['网元名称'] == x['所属基站']]
            if GD_sele.empty == True:
                return ('否')
            else:
                GD_sele.fillna('空空',inplace = True)
                is_GD = GD_sele.iloc[-1,-1]
                if is_GD == '空空':
                    return ('是，工单处理中')
                else:
                    Alert_time_s = Alert_sele.iloc[-1,0] # 告警表中告警开始时间
                    Alert_time = datetime.datetime.strptime(str(Alert_time_s), "%Y/%m/%d %H:%M:%S")
                    GD_sele['归档操作时间'] = GD_sele['归档操作时间'].map(lambda x:datetime.datetime.strptime(str(x),"%Y/%m/%d %H:%M:%S"))
                    if Alert_time > GD_sele.iloc[-1,-1]:  # 工单表中告警消除时间
                        return ('是，工单已完结({}),但仍有告警'.format(GD_sele.iloc[-1, -1]))
                    elif Alert_time <= GD_sele.iloc[-1,-1]:
                        return ('是，工单已完结({})'.format(GD_sele.iloc[-1,-1]))

        XJ_Ord['上周是否已派单'] = XJ_Ord.apply(CONcat,axis=1)
        def GDcat(x):
            GD_sele = GD_list_0[GD_list_0['网元名称'] == x['所属基站']]
            GD_sele.fillna('工单仍在处理中', inplace=True)
            if GD_sele.empty == True:
                return ('无')
            else:
                GD_sele['汇总'] = GD_sele.apply(lambda x:str(x['工单流水号'])+'_'+str(x['维护单位'])+'_'+str(x['故障发生时间'])+'_'+str(x['处理措施']),axis = 1)
                GD_contact = GD_sele.groupby(['网元名称'])['汇总'].apply(list).to_frame()
                GD_contact_o = GD_contact.iloc[0, 0]
                GD_final = '[' + ']|\n['.join(GD_contact_o) + ']'
                return GD_final
        XJ_Ord['历史工单记录'] = XJ_Ord.apply(GDcat,axis=1)

        XJ_Ord['工单完结时间'] = XJ_Ord['上周是否已派单'].map(lambda x:str((re.findall(re.compile(r'[(](.*?)[)]', re.S), x))[0]) if '工单已完结' in x else '*')
        Alert_list['告警开始时间'] = pd.to_datetime(Alert_list['告警开始时间'])
        Alert_list.sort_values(by=['告警开始时间'], inplace=True)
        def After_GD(x,data_alarm):
            BT_name = x['所属基站']
            data_alarm = data_alarm[data_alarm['网元名称'] == BT_name]
            if x['工单完结时间']=='*':
                return '无'
            else:
                x['工单完结时间'] = pd.to_datetime(x['工单完结时间'],format="%Y/%m/%d %H:%M:%S")
                data_alarm = data_alarm[data_alarm['告警开始时间'] >= x['工单完结时间']]
                alarm_pre = pd.DataFrame(data_alarm['告警名称'].value_counts())
                alarm_pre.reset_index(drop=False, inplace=True)
                alarm_pre.columns = ['告警名称', '发生次数']
                IF_alarmalso_list = []
                for n in range(len(alarm_pre)):
                    IF_alarmalso_list.append(alarm_pre.iloc[n, 0] + '：' + str(alarm_pre.iloc[n, 1]) + '次')
                IF_alarmalso = '，'.join(IF_alarmalso_list)
                return IF_alarmalso

        XJ_Ord['工单处理后仍发生告警频次']=XJ_Ord.apply(lambda x:After_GD(x,Alert_list),axis=1)
        XJ_Ord.drop(['工单完结时间'], axis=1, inplace=True)

        # 增加重点关注项目所属类别
        Alert_X1 = pd.read_excel('./Data/汇总/Project_Opt/告警分类-陕西.xlsx')
        Alert_Class = Alert_X1[['告警标题', '无线告警归类']]

        def Deal_Class(x):
            Alert_num = x.split(',')
            Alert_sum = []
            for Alert_i in Alert_num:
                Alert_name = Alert_i.split('：')[0].strip(' ')
                Alert_sum.append(Alert_name)
            Alert_pd = pd.DataFrame(np.array(Alert_sum), columns=['告警标题'])
            # Alert_pd.reset_index(inplace=True)
            Alert_meg = pd.merge(Alert_pd, Alert_Class, on='告警标题', how='left')
            Alert_meg.drop_duplicates(subset='无线告警归类', inplace=True)
            Alert_fb = Alert_meg['无线告警归类'].values.tolist()
            Alert_fb_jo = ('，').join(Alert_fb)
            return Alert_fb_jo

        XJ_Ord['重点关注项目所属类别'] = XJ_Ord['重点关注影响业务告警发生频次'].map(lambda x: Deal_Class(x))

        XJ_Ord = XJ_Ord[['巡检优先级','基站健康度异常程度','巡检重点关注项目','重点关注影响业务告警发生频次','重点关注项目所属类别','未来三天退服概率','所属基站','备注','所属机房','预约站时间','上周是否已派单','历史工单记录','工单处理后仍发生告警频次','连续问题天数','前1天优先级','前2天优先级','前3天优先级','告警时间戳']]

        save_path = r"./Data/{}/Inspect_List/{}/{}_清单_{}.csv".format(distr, ftype, distr, Inspect_date)
        XJ_Ord.to_csv(save_path, encoding='gbk', index=False)
        save_path1 = r"{}/{}_清单_{}.csv".format(para.XJ_result, distr, Inspect_date)
        if os.path.exists(para.XJ_result):
            XJ_Ord.to_csv(save_path1, encoding='gbk', index=False)
            print("清单生成完成，结果保存在: {}。".format(save_path1))
        else:
            print('ERROR:{} is not exists!'.format(para.XJ_result))


