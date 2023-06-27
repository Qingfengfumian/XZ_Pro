# -*- coding: utf-8 -*-
'''
@File  : Solution_suggestion_L3.py
@Author: CMDI_AI
@Date  : 2022/4/14 17:21
'''
import pandas as pd
import numpy as np
import re

'''
调用方式：from Solution_suggestion_L3 import process_del
df = process_del(df)
其中巡检清单中需要有'巡检重点关注项目'列
并将config文件夹放入所在工程目录
巡检清单df传入

'''


def to_participle(content):
    content = content.replace("\r\n", "")  # 删除换行
    content = content.replace("\\n", "")  # 删除换行
    content = content.replace(" ", "")  # 删除空行、多余的空格
    content = re.sub("[\s+\\!\_,$%^*(+\"\']+|[+——！？~@#￥%……&*（）]+", "", content)
    content = content.upper()
    return content


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


def Deal_Way(x, Alert_Way):
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
        pass
    Alert_fb_jo = ('\n').join(Alert_fb)
    return Alert_fb_jo


def process_del(df_inspection_list):
    df_inspection_list['关注项目_保留'] = df_inspection_list['巡检重点关注项目'].map(lambda x: Deal_retain(x))
    # Alert_X = pd.read_excel('config/配置文档3：处理方案规则配置页面--告警处理建议映射.xlsx', sheet_name=None)
    # print(Alert_X.keys())
    Alert_X1 = pd.read_excel('config/配置文档3：处理方案规则配置页面--告警处理建议映射.xlsx', sheet_name='华为')
    Alert_X2 = pd.read_excel('config/配置文档3：处理方案规则配置页面--告警处理建议映射.xlsx', sheet_name='中兴')
    Alert_X = pd.concat([Alert_X1, Alert_X2])
    Alert_Way = Alert_X[['告警标题', '处理方案建议']]
    df_inspection_list['告警处理措施'] = df_inspection_list['关注项目_保留'].map(lambda x: Deal_Way(x, Alert_Way))
    df_inspection_list['告警处理措施1'] = df_inspection_list['告警处理措施'].apply(to_participle)
    df_inspection_list.drop(['关注项目_保留'], axis=1, inplace=True)
    return df_inspection_list


if __name__ == '__main__':
    data_inspection_list = pd.read_csv('西安_清单_20220412_4G_华为.csv', encoding='gbk')

    df_inspection_list = process_del(data_inspection_list)
