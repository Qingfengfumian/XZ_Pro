"""
editor:wyn
time:20210120
version:2.01
"""
import os
import time
import datetime
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
def sep_file(file_path):
    start_time = time.time()  # 程序开始时间
    print("开始拆分文件: %s" % (datetime.datetime.fromtimestamp(start_time)))
    with open(file_path,'r',encoding='gbk') as f:
        paragraph_cnt = 0
        paragraphs = []
        paragraph = []
        for line_num,text in enumerate(f.readlines(),start=1):
            last_paragraph_cnt = paragraph_cnt #1
            if re.search('(===)+?',text):
                paragraph_cnt += 1 #2
                if (paragraph_cnt>last_paragraph_cnt):
                    para_list = ''.join(paragraph)
                    paragraphs.append(para_list)
                paragraph = []
                continue
            else:
                paragraph.append(text)
    success_paras = [s.replace('\t', '') for s in paragraphs if '+++' in s]
    print("完成拆分文件: %s" % (datetime.datetime.fromtimestamp(start_time)))
    return success_paras
def decode_omc(success_paras0,type_name):
    col = ['基站名称', '指令名称', '执行时间']
    bs_cnt, i ,k= 0, 0,0
    omc_dict, detail_dict = {}, {}
    start, end, pass_flag, single, double_page, add_channel = False, False, False, False, False, False
    for para_num,success_para in tqdm(enumerate(success_paras0,start=1)):
        texts = success_para.split('\n')
        for line in texts:
            if line != '\n':line = line.replace('\n', '')
            if line == '':continue
            if end == True:  # 完成一个基站解析,刷新参数
                detail_dict, i, k,pass_flag, single = {}, 0, 0,False, False
                end = False
            if '+++' in line:
                enb_name = line.split()[1]
                detail_dict[col[2]] = ' '.join(line.split()[2:]) #cmd_time
            if '查询单板温度' in line:detail_dict[col[1]] = 'DSP BRDTEMP:;'
            if '查询光电模块动态信息' in line:detail_dict[col[1]] = 'DSP SFP:;'
            if '查询BBU电源板状态' in line:detail_dict[col[1]] = 'DSP UPEU:;'
            if '查询驻波比测试结果' in line:detail_dict[col[1]] = 'DSP VSWR:;'
            if '端口通道信息' in line:add_channel = True
            if '柜号' in line and '=' in line:#单柜
                single = True
                line = line.replace(" ", "")
                detail_dict.setdefault('info_0', {})[line.split('=')[0]] = line.split('=')[1]
                continue
            if '柜号' in line and '=' not in line: #非单柜
                if double_page == False: #单页
                    dic_cols = line.split()
                    single = False
                if add_channel:
                    channel_cols = line.split()
                continue
            if single == True:
                if '=' in line and '结果' not in line:
                    line = line.replace(" ", "")
                    detail_dict.setdefault('info_0', {})[line.split('=')[0]] = line.split('=')[1]
            if single == False:
                if re.search('^\d+\\s+\d+',line):
                    # 替换SFP的厂家名称
                    line = line.replace('HG GENUINE','HG-GENUINE')
                    line = line.replace('FINISAR CORP.','FINISAR-CORP.')
                    line = line.replace('EOPTOLINK INC','EOPTOLINK-INC')
                    line = line.replace('ALLRAY INC.','ALLRAY-INC.')
                    values = line.split()
                    if add_channel:
                        for j in range(len(channel_cols)):
                            detail_dict.setdefault('info_channel_%s'% k, {})[channel_cols[j]] = values[j]
                        k += 1
                    else:
                        for j in range(len(dic_cols)):
                            detail_dict.setdefault('info_%s' % i, {})[dic_cols[j]] = values[j]
                        i += 1
            if '仍有后续报告输出' in line:
                double_page = True
                continue
            if '共有' in line: #双页输出完毕
                double_page = False
                continue
            if '---    END' in line and double_page==False:
                end = True
                col_list = []
                add_channel = False
                if 'info_0' in detail_dict.keys():
                    bs_cnt += 1
                    omc_dict.setdefault(enb_name, detail_dict)
    print('%s共解析段落<%s>\n共解析基站%s'%(type_name,len(success_paras0),bs_cnt))
    return omc_dict,bs_cnt
def mk_csv(omc_dict,omc_type):
    start_time = time.time()  #
    print("开始生成csv: %s" % (datetime.datetime.fromtimestamp(start_time)))
    for key in omc_dict.keys():
        dic_cols = list(omc_dict[key]['info_0'].keys())
        break
    if omc_type == 'SFP':
        dic_cols.extend(['通道号', '通道可用状态'])
    base_col = ['基站名称', '指令名称', '执行时间']
    base_col.extend(dic_cols)
    ColumnNum = np.array(base_col)
    len_Column = len(ColumnNum)  # col_cnt
    key_num = 0
    OMC_dict = []
    for key, values in omc_dict.items():
        Col_lst = ['空' for _ in range(len_Column)]
        for key1, values1 in values.items():
            Col_lst[0] = key
            if 'info' not in key1:
                ls_idx = base_col.index(key1)  # exist title
                Col_lst[ls_idx] = values1
            else:
                for key2, values2 in values1.items():
                    ls_idx = base_col.index(key2)  # exist title
                    Col_lst[ls_idx] = values2
                temp = Col_lst.copy()
                OMC_dict.append(temp)
        key_num += 1
    OMC_np = np.array(OMC_dict)
    OMC_df = pd.DataFrame(OMC_np, columns=base_col)
    if omc_type == 'SFP':
        OMC_channel = OMC_df[~(OMC_df['通道号'] == '空')]
        OMC_channel = OMC_channel[
            ['基站名称', '指令名称', '执行时间', '柜号', '框号', '槽号', '端口类型', '端口号', '通道号', '发送光功率(0.1微瓦)', '接收光功率(0.1微瓦)',
             '发送光功率(0.01毫瓦分贝)', '接收光功率(0.01毫瓦分贝)', '电流(2微安培)', '通道可用状态']]
        OMC_org = OMC_df[OMC_df['通道号'] == '空'].drop(['通道号', '通道可用状态'], axis=1)
        OMC_DF = pd.merge(OMC_org, OMC_channel, how='left',
                          on=['基站名称', '指令名称', '执行时间', '柜号', '框号', '槽号', '端口类型', '端口号'])
        COL = [column for column in OMC_DF]
        NO_C_COL = [c if '_y' in c else 0 for c in COL]
        while 0 in NO_C_COL:
            NO_C_COL.remove(0)
        C_COL = [c if '_x' in c else 0 for c in COL]
        while 0 in C_COL:
            C_COL.remove(0)

        def reset_col(df):
            NEW_COL = [column for column in df]
            NEW_C_COL = [str(c).replace('_y', '').replace('_x', '') if ('_y' in c) or ('_x' in c) else c for c in
                         NEW_COL]
            df.columns = NEW_C_COL
            return df

        NO_CHANNEL = OMC_DF[OMC_DF['通道号'].isnull()].drop(NO_C_COL, axis=1)
        NO_CHANNEL = reset_col(NO_CHANNEL)
        CHANNEL = OMC_DF[OMC_DF['通道号'].notnull()].drop(C_COL, axis=1)
        CHANNEL = reset_col(CHANNEL)
        OMC_res = pd.concat([CHANNEL, NO_CHANNEL])
        COL = [column for column in CHANNEL]
        COL.remove('通道号')
        COL.insert(8, '通道号')
        OMC_df = OMC_res[COL]
    return OMC_df

def del_csv(OMC_df,type):
    """待修改"""
    start_time = time.time()  # 程序开始时间
    print("开始处理CSV: %s" % (datetime.datetime.fromtimestamp(start_time)))
    OMC_csv = OMC_df.copy()
    if type=='BRDTEMP':
        temp_1 = OMC_csv[~OMC_csv['单板温度(℃)'].str.contains("NULL")]
        print('****单板温度缺失:%s条' % (OMC_csv.shape[0]-temp_1.shape[0]))
        mean_1 = temp_1['单板温度(℃)'].map(lambda x:int(x)).mean()
        temp_2 = OMC_csv[~OMC_csv['功放温度(℃)'].str.contains("NULL")]
        print('****功放温度缺失:%s条' % (OMC_csv.shape[0]-temp_2.shape[0]))
        mean_2 = temp_2['功放温度(℃)'].map(lambda x:int(x)).mean()
        OMC_csv['单板温度(℃)'] = OMC_csv['单板温度(℃)'].map(lambda x:str(int(mean_1)) if x=='NULL' else x)
        OMC_csv['功放温度(℃)'] = OMC_csv['功放温度(℃)'].map(lambda x:str(int(mean_2)) if x=='NULL' else x)
    elif type=='SFP':
        col_name = OMC_csv.columns[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,  21,22,  30,31]].tolist()
        OMC_csv = OMC_csv[col_name].replace('NULL',np.nan)
        org_size = OMC_csv.shape[0]
        OMC_csv = OMC_csv.dropna(subset= col_name[-4:],how='any',axis='index')
        print('****去除空值:%s条'%(org_size-OMC_csv.shape[0]))
    elif type=='UPEU':
        col_name = OMC_csv.columns[[0, 1, 2, 3, 4, 5, 8]].tolist()
        OMC_csv = OMC_csv[col_name].replace('NULL',np.nan)
        org_size = OMC_csv.shape[0]
        OMC_csv = OMC_csv.dropna(subset = [col_name[-1]],axis='index')
        print('****去除空值:%s条'%(org_size-OMC_csv.shape[0]))
    elif type=='VSWR':
        OMC_csv = OMC_csv.replace('NULL',np.nan)
        org_size = OMC_csv.shape[0]
        OMC_csv = OMC_csv.dropna(subset=[OMC_csv.columns[-1]])
        print('****去除空值:%s条'%(org_size-OMC_csv.shape[0]))
    # OMC_csv.to_csv(save_path,index=None)
    end_time = time.time()  # 程序结束时间
    print('结束处理CSV: %s' % (datetime.datetime.fromtimestamp(end_time)))
    total_time = end_time - start_time
    print('共计%.2f秒 ' % (total_time))
    return OMC_csv
if __name__ == '__main__':
    file_path = 'E:\AItask\AIOps_ShanX\战略项目\OMC指令解析/MMLTask_省维护定时任务_20201123_195007.txt'
    save_path = 'E:\AItask\AIOps_ShanX\战略项目\OMC指令解析/OMC指令解析结果v2/'
    start_time = time.time()  #
    ALL_PARA = sep_file(file_path)
    # print('拆分类型:单板温度、光电模块动态信息、BBU电源板状、查询驻波比测试结果')
    title_list = ['查询单板温度','查询光电模块动态信息','查询BBU电源板状态','查询驻波比测试结果']
    type_list = ['BRDTEMP','SFP','UPEU','VSWR']
    for type_name,type_id in zip(title_list, type_list):
        spt_data = [s for s in ALL_PARA if type_name in s]
        omc_dict,bs_cnt = decode_omc(spt_data,type_name)
        OMC_DF = mk_csv(omc_dict,omc_type=type_id)

        OMC_DF.to_csv(os.path.join(save_path,'RES_%s.csv'%type_id),encoding='gbk',index=False)
    end_time = time.time()  #
    total_time = end_time - start_time
    print('整体用时：%s' % (total_time))
