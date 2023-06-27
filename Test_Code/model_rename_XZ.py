import os
import shutil

# ------------------------- 20220310退服预测模型修改 ---------------------
# ①改地市清单 ②改制式 ③改厂家 ④改日期
Verdom_N = '华为'
E_Verdom = 'HW'
AD_Verdom = '_{}'.format(E_Verdom)
date = '20220501'
date_high = '20220710'
dist_list = ['拉萨'] #,
# 4G 华为 # ['拉萨','昌都','山南','林芝']
# 5G 华为 # ['拉萨','昌都','山南','林芝']
# 4G 中兴 # ['日喀则','那曲','阿里']
# 5G 中兴 # ['日喀则','那曲','阿里']
for dist_name in dist_list:
    # Name_ = 'YL'
    org_path = '/home/XZ_Pro_AIOps/TR_TF_code/XZ_5G_HW/LS_5G_HW/materials/'
    path = '/home/XZ_Pro_AIOps/Data/{}/Inter_Result/5G/'.format(dist_name)

    # date = '20210506'
    # date_high = '20210930'
    #
    # ----------- 移动文件 --------------
    for i in os.listdir(org_path):
        print(i)
        # if (Name_ in i):
        if ('.pkl' in i):
            source = org_path + i
        # source = '/home/weihu/SX_Project/ShanX_CC/CC_Data/西安/read_data/告警标题.txt'
            target = path + '/{}_退服_模型/'.format(Verdom_N)
            if not os.path.exists(target):
                os.makedirs(target)
            shutil.copy(source, target)
        elif ('.txt' in i):
            if ('FRule' not in i):
                source = org_path + i
                # source = '/home/weihu/SX_Project/ShanX_CC/CC_Data/西安/read_data/告警标题.txt'
                target = path + '/{}_特征标题/'.format(Verdom_N)
                if not os.path.exists(target):
                    os.makedirs(target)
                shutil.copy(source, target)

        elif ('.json' in i):
            source = org_path + i
            # source = '/home/weihu/SX_Project/ShanX_CC/CC_Data/西安/read_data/告警标题.txt'
            target = path + '/{}_特征编码/'.format(Verdom_N)
            if not os.path.exists(target):
                os.makedirs(target)
            shutil.copy(source, target)

    target = path + '/{}_退服_模型/'.format(Verdom_N)
    source1 = '/home/XZ_Pro_AIOps/TR_TF_code' + '/FRule_全省_20200701-20210228.txt'
    shutil.copy(source1, target)
    gaiming1 = target + '/FRule_全省_20200701-20210228.txt'
    hou1 = target + '/FRule_{}_20200701-20210228.txt'.format(dist_name)
    os.rename(gaiming1, hou1)

for dist_name in dist_list:
    path = '/home/XZ_Pro_AIOps/Data/{}/Inter_Result/5G/'.format(dist_name)
    Name_ = 'QS'
    wenjian1 = '华为_特征编码'
    gaiming1 = '{}/{}/{}_{}-{}_5G_HW_NoList_Del6D_labelencoder.json'.format(path,wenjian1,Name_,date,date_high)
    gaiming2 = '{}/{}/{}_{}-{}_5G_HW_NoList_Del7D_labelencoder.json'.format(path,wenjian1,Name_,date,date_high)
    hou1 = '{}/{}/labelencoder_del6_{}-{}5G.json'.format(path,wenjian1,date,date_high)
    hou2 = '{}/{}/labelencoder_del7_{}-{}5G.json'.format(path,wenjian1,date,date_high)
    try:
        os.rename(gaiming1,hou1)
        os.rename(gaiming2,hou2)
    except:print('{}{}'.format(dist_name,wenjian1))
    wenjian2 = '华为_特征标题'
    gaiming3 = '{}/{}/{}_{}-{}_5G_HW_NoList_Del6D_input_features.txt'.format(path,wenjian2,Name_,date,date_high)
    gaiming4 = '{}/{}/{}_{}-{}_5G_HW_NoList_Del7D_input_features.txt'.format(path,wenjian2,Name_,date,date_high)
    hou3 = '{}/{}/input_features_del6_{}-{}5G.txt'.format(path,wenjian2,date,date_high)
    hou4 = '{}/{}/input_features_del7_{}-{}5G.txt'.format(path,wenjian2,date,date_high)
    try:
        os.rename(gaiming3,hou3)
        os.rename(gaiming4,hou4)
    except:print('{}{}'.format(dist_name,wenjian2))

    wenjian3 = '华为_退服_模型'
    for i in range(1,6):
        gaiming5 = '{}/{}/lgb_model_{}_{}-{}_5G_HW_NoList_Del6D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
        gaiming6 = '{}/{}/lgb_model_{}_{}-{}_5G_HW_NoList_Del7D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
        hou5 = '{}/{}/lgb_model_del6_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
        hou6 = '{}/{}/lgb_model_del7_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
        gaiming7 = '{}/{}/xgb_model_{}_{}-{}_5G_HW_NoList_Del6D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
        gaiming8 = '{}/{}/xgb_model_{}_{}-{}_5G_HW_NoList_Del7D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
        hou7 = '{}/{}/xgb_model_del6_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
        hou8 = '{}/{}/xgb_model_del7_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
        try:
            os.rename(gaiming5, hou5)
            os.rename(gaiming6, hou6)
        except:
            print('{}{}'.format(dist_name, wenjian3))
        try:
            os.rename(gaiming7, hou7)
            os.rename(gaiming8, hou8)
        except:
            print('{}{}'.format(dist_name, wenjian3))

# # ------------------------- 20220310异常检测模型修改 ---------------------
# # ①改地市清单 ②改制式 ③改厂家 ④改
# Verdom_N = '华为'
# E_Verdom = 'HW'
# AD_Verdom = '_{}'.format(E_Verdom)
# dist_list = ['拉萨','昌都','山南','林芝']
# # 4G 华为 # ['拉萨','昌都','山南','林芝']
# # 5G 华为 # ['拉萨','昌都','山南','林芝']
# # 4G 中兴 # ['日喀则','那曲','阿里']
# # 5G 中兴 # ['日喀则','那曲','阿里']
# for dist_name in dist_list:
#     # Name_ = 'YL'
#     org_path = '/home/XZ_Pro_AIOps/Data/全省/Inter_Result/4G/'
#     # org_path = '/home/AIOps/LN_Project/TR_TF_code/LN_4G_ZX/QS_4G_ZX/materials/'
#     path = '/home/XZ_Pro_AIOps/Data/{}/Inter_Result/4G/'.format(dist_name)
#
#     # date = '20210506'
#     # date_high = '20210930'
#     #
#     # ----------- 移动文件 --------------
#     source1 = org_path + '/{}_故障_标准化/'.format(Verdom_N)
#     for i in os.listdir(source1):
#         source_1 = source1+i
#         target = path + '/{}_故障_标准化/'.format(Verdom_N)
#         if not os.path.exists(target):
#             os.makedirs(target)
#         shutil.copy(source_1, target)
#
#     source2 = org_path + '/{}_告警_标题/'.format(Verdom_N)
#     for i in os.listdir(source2):
#         source_2 = source2 + i
#         target = path + '/{}_告警_标题/'.format(Verdom_N)
#         if not os.path.exists(target):
#             os.makedirs(target)
#         shutil.copy(source_2, target)
#
#     source3 = org_path + '/{}_故障_模型/'.format(Verdom_N)
#     for i in os.listdir(source3):
#         source_3 = source3 + i
#         target = path + '/{}_故障_模型/'.format(Verdom_N)
#         if not os.path.exists(target):
#             os.makedirs(target)
#         shutil.copy(source_3, target)
# # ------------------------- END 20220310异常检测模型修改 ---------------------

    # target = path + '/{}_退服_模型/'.format(Verdom_N)
    # source1 = '/home/AIOps/LN_Project/TR_TF_code' + '/FRule_全省_20200701-20210228.txt'
    # shutil.copy(source1, target)
    # gaiming1 = target + '/FRule_全省_20200701-20210228.txt'
    # hou1 = target + '/FRule_{}_20200701-20210228.txt'.format(dist_name)
    # os.rename(gaiming1, hou1)

# for dist_name in dist_list:
#     path = '/home/AIOps/LN_Project/Data/{}/Inter_Result/5G/'.format(dist_name)
#     Name_ = 'QS'
#     date = '20211008'
#     date_high = '20220213'
#     wenjian1 = '中兴_特征编码'
#     gaiming1 = '{}/{}/{}_{}-{}_5G_ZX_NoList_Del6D_labelencoder.json'.format(path,wenjian1,Name_,date,date_high)
#     gaiming2 = '{}/{}/{}_{}-{}_5G_ZX_NoList_Del7D_labelencoder.json'.format(path,wenjian1,Name_,date,date_high)
#     hou1 = '{}/{}/labelencoder_del6_{}-{}5G.json'.format(path,wenjian1,date,date_high)
#     hou2 = '{}/{}/labelencoder_del7_{}-{}5G.json'.format(path,wenjian1,date,date_high)
#     try:
#         os.rename(gaiming1,hou1)
#         os.rename(gaiming2,hou2)
#     except:print(11)
#     wenjian2 = '中兴_特征标题'
#     gaiming3 = '{}/{}/{}_{}-{}_5G_ZX_NoList_Del6D_input_features.txt'.format(path,wenjian2,Name_,date,date_high)
#     gaiming4 = '{}/{}/{}_{}-{}_5G_ZX_NoList_Del7D_input_features.txt'.format(path,wenjian2,Name_,date,date_high)
#     hou3 = '{}/{}/input_features_del6_{}-{}5G.txt'.format(path,wenjian2,date,date_high)
#     hou4 = '{}/{}/input_features_del7_{}-{}5G.txt'.format(path,wenjian2,date,date_high)
#     try:
#         os.rename(gaiming3,hou3)
#         os.rename(gaiming4,hou4)
#     except:print(11)
#
#     wenjian3 = '中兴_退服_模型'
#     for i in range(1,6):
#         gaiming5 = '{}/{}/lgb_model_{}_{}-{}_5G_ZX_NoList_Del6D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
#         gaiming6 = '{}/{}/lgb_model_{}_{}-{}_5G_ZX_NoList_Del7D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
#         hou5 = '{}/{}/lgb_model_del6_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
#         hou6 = '{}/{}/lgb_model_del7_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
#         gaiming7 = '{}/{}/xgb_model_{}_{}-{}_5G_ZX_NoList_Del6D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
#         gaiming8 = '{}/{}/xgb_model_{}_{}-{}_5G_ZX_NoList_Del7D_{}.pkl'.format(path, wenjian3, Name_,date,date_high,i)
#         hou7 = '{}/{}/xgb_model_del6_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
#         hou8 = '{}/{}/xgb_model_del7_{}_{}-{}5G.pkl'.format(path, wenjian3,i,date,date_high)
#         try:
#             os.rename(gaiming5, hou5)
#             os.rename(gaiming6, hou6)
#         except:
#             print(11)
#         try:
#             os.rename(gaiming7, hou7)
#             os.rename(gaiming8, hou8)
#         except:
#             print(11)
