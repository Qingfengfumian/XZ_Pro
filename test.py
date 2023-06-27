import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

# warnings.filterwarnings('ignore')


def run_effect_eva(Org_path, Factory_C, City_name, mode_name, Org_path_pro, City_name_pro, pre_ds):
    city_result_6D = pd.read_csv(
        '{}/{}_{}_{}_Res_NoList_delGC6D-{}R.csv'.format(Org_path, City_name, mode_name, Factory_C, pre_ds),
        encoding='gbk')
    pro_result_6D = pd.read_csv(
        '{}/{}_{}_{}_Res_NoList_delGC6D-{}R.csv'.format(Org_path_pro, City_name_pro, mode_name, Factory_C, pre_ds),
        encoding='gbk')

    city_result_6D['基站日期index'] = city_result_6D.apply(lambda x: str(x['基站id']) + "-" + x['date'], axis=1)
    pro_result_6D['基站日期index'] = pro_result_6D.apply(lambda x: str(x['基站id']) + "-" + x['date'], axis=1)

    city_merge_6D = pd.merge(city_result_6D, pro_result_6D[['基站日期index', 'pred_probability']].rename(
        columns={'pred_probability': '全省概率'}), on='基站日期index')

    prbi_list = []
    for probili in range(35, 85, 5):
        city_merge_result = city_merge_6D[city_merge_6D['pred_probability'] >= probili / 100]
        if len(city_merge_result[['Label_5']]) == 0:
            prbi_list.append([probili / 100, 0, 0, 0])
            continue
        prec = city_merge_result['Label_5'].sum() / len(city_merge_result[['Label_5']])
        prbi_list.append([probili / 100, len(city_merge_result[['Label_5']]), city_merge_result['Label_5'].sum(), prec])
    prbi_list_pd = pd.DataFrame(prbi_list)
    prbi_list_pd.columns = ['阈值', '地市总基站', '地市退服基站', '地市预测精确率']

    prbi_list_1 = []
    for probili in range(35, 85, 5):
        city_merge_result = city_merge_6D[city_merge_6D['全省概率'] >= probili / 100]
        if len(city_merge_result[['Label_5']]) == 0:
            prbi_list_1.append([probili / 100, 0, 0, 0])
            continue
        prec = city_merge_result['Label_5'].sum() / len(city_merge_result[['Label_5']])
        prbi_list_1.append(
            [probili / 100, len(city_merge_result[['Label_5']]), city_merge_result['Label_5'].sum(), prec])
    prbi_list_1_pd = pd.DataFrame(prbi_list_1)
    prbi_list_1_pd.columns = ['阈值', '全省总基站', '全省退服基站', '全省预测精确率']

    prbi_sum_pd = pd.merge(prbi_list_pd, prbi_list_1_pd, on='阈值')
    prbi_sum_pd.to_csv('{}/{}_{}_{}_6D_compare-{}R.csv'.format(Org_path, City_name, mode_name, Factory_C, pre_ds),
                       index=False, encoding='gbk')
    print('6D验证完毕')

#     city_result_7D = pd.read_csv('{}/{}_{}_{}_Res_NoList_delGC7D.csv'.format(Org_path, City_name, mode_name, Factory_C),
#                                  encoding='gbk')
#     pro_result_7D = pd.read_csv(
#         '{}/{}_{}_{}_Res_NoList_delGC7D.csv'.format(Org_path_pro, City_name_pro, mode_name, Factory_C), encoding='gbk')

#     city_result_7D['基站日期index'] = city_result_7D.apply(lambda x: str(x['基站id']) + "-" + x['date'], axis=1)
#     pro_result_7D['基站日期index'] = pro_result_7D.apply(lambda x: str(x['基站id']) + "-" + x['date'], axis=1)

#     city_merge_7D = pd.merge(city_result_7D, pro_result_7D[['基站日期index', 'pred_probability']].rename(
#         columns={'pred_probability': '全省概率'}), on='基站日期index')

#     prbi_list = []
#     for probili in range(35, 85, 5):
#         city_merge_result = city_merge_7D[city_merge_7D['pred_probability'] >= probili / 100]
#         if len(city_merge_result[['Label_5']]) == 0:
#             prbi_list.append([probili / 100, 0, 0, 0])
#             continue
#         prec = city_merge_result['Label_5'].sum() / len(city_merge_result[['Label_5']])
#         prbi_list.append([probili / 100, len(city_merge_result[['Label_5']]), city_merge_result['Label_5'].sum(), prec])
#     prbi_list_pd = pd.DataFrame(prbi_list)
#     prbi_list_pd.columns = ['阈值', '地市总基站', '地市退服基站', '地市预测精确率']

#     prbi_list_1 = []
#     for probili in range(35, 85, 5):
#         city_merge_result = city_merge_7D[city_merge_7D['全省概率'] >= probili / 100]
#         if len(city_merge_result[['Label_5']]) == 0:
#             prbi_list_1.append([probili / 100, 0, 0, 0])
#             continue
#         prec = city_merge_result['Label_5'].sum() / len(city_merge_result[['Label_5']])
#         prbi_list_1.append(
#             [probili / 100, len(city_merge_result[['Label_5']]), city_merge_result['Label_5'].sum(), prec])
#     prbi_list_1_pd = pd.DataFrame(prbi_list_1)
#     prbi_list_1_pd.columns = ['阈值', '全省总基站', '全省退服基站', '全省预测精确率']

#     prbi_sum_pd = pd.merge(prbi_list_pd, prbi_list_1_pd, on='阈值')
#     prbi_sum_pd.to_csv('{}/{}_{}_{}_7D_compare.csv'.format(Org_path, City_name, mode_name, Factory_C), index=False, encoding='gbk')
#     print('7D验证完毕')

