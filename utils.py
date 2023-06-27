import os
import argparse
import json
import datetime
import multiprocessing
# import pymysql
import pandas as pd

# def mysql_dealdata(table_name,key_value,IF_multi):
#     # db = pymysql.connect(host='10.206.161.67',user='zhyw@123',password='ZY@!3415xd',db='ai_paltform',port=3306)
#     db = pymysql.connect(host='localhost', user='root', password='root', db='ai_paltform', port=3306)
#     ident_config = "select * from {}".format(table_name)
#     df_config = pd.read_sql(ident_config, con=db)
#     mysql_config = df_config[df_config['config_code'] == key_value]['config_value'].values[0]
#     # print(mysql_config)
#     if IF_multi == 0:
#         return mysql_config
#     else:
#         json_str = json.loads(mysql_config)
#         mysql_values = json_str['values']
#         mysql_titles = json_str['titles']
#
#         df_values = pd.DataFrame(mysql_values)
#         df_titles = pd.DataFrame(mysql_titles)
#
#         df_columns = df_titles['title'].values.tolist()
#         for df_i in range(len(df_titles)):
#             columns_name = "name{}".format(df_i)
#             df_values[columns_name] = df_values['value'].map(lambda x:x.split(";")[df_i])
#
#         df_values.drop('value',inplace=True,axis=1)
#         df_values.columns = df_columns
#         return df_values
#     db.close()

def mkdir(path):
    path = path.strip()
    path = path.rstrip("//") # 删除 string 字符串末尾的指定字符
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
def find_new_file2(dir,day):
    '''查找目录下最新的文件 文件命名：xxx_日期1-日期2.xxx'''
    files = os.listdir(dir)
    file_dict = {}
    for file in files:
        key_mode = "_del{}_".format(day)
        if key_mode in file:
            file_date = file.split('_')[-1].split('.')[0]
            new_date = file_date.split('-')[-1]  # get date
            file_dict[file] = new_date
    file_name = max(file_dict, key=file_dict.get)
    return file_name

def find_new_file(dir):
    '''查找目录下最新的文件'''
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + "//" + fn)
    if not os.path.isdir(dir + "//" + fn) else 0)
    file = os.path.join(dir, file_lists[-1])
    return file

def find_new_model_file(dir, modtype):
    files = os.listdir(dir)
    file_dict = {}
    for file in files:
        if (file.split('_')[0] == modtype):
            file_date = file.split('_')[-1].split('.')[0]
            new_date = file_date.split('-')[-1]  # get date
            file_dict[file] = new_date
    file_name = max(file_dict, key=file_dict.get)
    return file_name
def load_model_infos(file_path):
    list_info = []
    with open(file_path, "r", encoding='utf-8') as f:
        for item in f.readlines():
            list_info.append(item.replace("\n", ""))
    return list_info
def save_model_infos(list_info, file_path):
    with open(file_path, "w",encoding='utf-8') as f:
        for item in list_info:
            f.write(item+"\n")
def load_from_json(file_path):
    with open(file_path, "r") as json_f:
        obj = json.load(json_f)
    return obj
def save_to_json(obj, file_path):
    with open(file_path, "w") as json_f:
        json.dump(obj, json_f)

real_warns_4G = ["[衍生告警]动环告警导致4G基站退服","小区不可用告警","网元连接中断","[衍生告警]同一基站发生所有LTE小区不可用告警","[衍生告警]同一基站发生多条LTE小区不可用告警","[衍生告警]中兴部分LTE小区退服", "LTE小区退出服务","基站退出服务", "RRU链路断", "网元断链告警"]
real_warns_5G = ["NR小区不可用告警","gNodeB退服告警","[衍生告警]华为5G基站退服","网元连接中断","[衍生告警]动环告警导致5G基站退服","[衍生告警]同一基站发生多条NR小区不可用告警","DU小区退服", "基站DU退服", "RRU链路断", "网元链路断", "[衍生关联]中兴部分NR小区退服", "[衍生告警]中兴5G基站退服"] # 华为 中兴
# 华为
# real_warns = ["[衍生告警]动环告警导致4G基站退服","小区不可用告警","网元连接中断","[衍生告警]同一基站发生所有LTE小区不可用告警","[衍生告警]同一基站发生多条LTE小区不可用告警"]  #4G华为
# real_warns = ["NR小区不可用告警","gNodeB退服告警","[衍生告警]华为5G基站退服","网元连接中断","[衍生告警]动环告警导致5G基站退服","[衍生告警]同一基站发生多条NR小区不可用告警"] # 5G华为
# 中兴
# real_warns = ["[衍生告警]中兴部分LTE小区退服", "LTE小区退出服务","基站退出服务", "RRU链路断", "网元断链告警"]  # 4G中兴
# real_warns = ["DU小区退服", "基站DU退服", "RRU链路断", "网元链路断", "[衍生关联]中兴部分NR小区退服", "[衍生告警]中兴5G基站退服"] # 5G中兴

# 4G 华为 # ['拉萨','昌都','山南','林芝']
# 5G 华为 # ['拉萨','昌都','山南','林芝']
# 4G 中兴 # ['日喀则','那曲','阿里']
# 5G 中兴 # ['日喀则','那曲','阿里']


def params_setup():
    parser = argparse.ArgumentParser()
    # ----------Common Para
    parser.add_argument('--distr_list', type=list, default=['拉萨','昌都','山南','林芝','日喀则','那曲','阿里'])
    # parser.add_argument('--distr_list', type=list, default=['拉萨'])
    parser.add_argument('--ftype_list', type=list, default=['华为'])#,'中兴'
    # parser.add_argument('--mode', type=str, default="train_XJ")  # 指定训练模式:train_TF\train_XJ\predict
    parser.add_argument('--mode', type=str, default="predict")  # 指定训练模式:train_TF\train_XJ\predict
    parser.add_argument('--ftype', type=str, default='华为') #指定参与训练预测的厂家
    parser.add_argument('--distr', type=str, default='铜川') # 指定退服模型储存地市
    # ----------Para for function 告警/工单/巡检结果保存路径

    parser.add_argument('--alarm_path', type=str, default='E:/PycharmProjects\XZ_Pro\服务器数据\告警数据/')  # 指定告警文件路径
    parser.add_argument('--ord_path', type=str, default='./Data/汇总/Fault_Ord/')  # 指定工单文件路径
    parser.add_argument('--XJ_result', type=str, default='./Data/Inspect_List/')  # 指定巡检结果保存路径
    parser.add_argument('--preserve_path', type=str, default='./Data/汇总/Reserve_List/')  # 预约站清单路径
    parser.add_argument('--out_path', type=str, default='./Data/Output_List/')  #
    # parser.add_argument('--out_path_web', type=str, default='./Data/Output_List/')  #
    parser.add_argument('--out_path_web', type=str, default='/data/resources/result/')
    parser.add_argument('--out_path_pre', type=str, default='/data/resources/result_predict/')
    # parser.add_argument('--out_path_pre', type=str, default='./Data/Output_List/')

    # parser.add_argument('--alarm_path', type=str, default='/data/Alarm_deal_data/')  # 指定告警文件路径
    # parser.add_argument('--ord_path', type=str, default='/home/cmdidata/order/everyday/')  # 指定工单文件路径
    # parser.add_argument('--XJ_result', type=str, default='/home/data/XJ_list/')  # 指定巡检结果保存路径
    # parser.add_argument('--preserve_path', type=str, default='/home/cmdidata/XJ_list/')  # 指定巡检结果保存路径
    # parser.add_argument('--out_path', type=str, default='/home/data/Output_List/')  # 指定巡检结果保存路径

    # ----------Para for function : data_del
    parser.add_argument('--sloc', type=list, default=['级别','告警'])  # 铜川:40 汉中:1500
    parser.add_argument('--grp_cnt', type=int, default=1500)  # 铜川:40 汉中:1500
    parser.add_argument('--del_alm', type=str, default=[])
        # ['用户面承载链路故障告警','X2接口故障告警','时钟参考源异常告警','小区服务能力下降告警']) # 铜川 汉中=[]
    parser.add_argument('--del_min3', type=str, default=\
        ['用户面承载链路故障告警', 'X2接口故障告警','License试运行告警'])  # 铜川 = 汉中

    # ----------Para for function : run_train_XJ
    parser.add_argument('--train_date', type=str, default="20220208-20220731")  # 指定训练数据时间

    # ----------Para for function : run_predict_XJ and run_predict_TF
    predict_time_0 = datetime.datetime.now() - datetime.timedelta(days=1) # 4-27
    predict_time_1 = predict_time_0 - datetime.timedelta(days=6)
    predict_time_high = str(predict_time_0).split(' ')[0].replace('-', '')
    predict_time_low = str(predict_time_1).split(' ')[0].replace('-', '')
    predict_time = predict_time_low + '-' + predict_time_high
    parser.add_argument('--date', type=str, default=predict_time)  # 指定预测数据时间 注意需要大于5周，否则交叉验证报错
    # parser.add_argument('--date', type=str, default="20220501-20220731")  # 指定预测数据时间 注意需要大于5周，否则交叉验证报错
    # parser.add_argument('--date', type=str, default="20220627-20220703")  # 指定预测数据时间 注意需要大于5周，否则交叉验证报错

    # ----------Para for function : run_predict_XJ
    parser.add_argument('--train_date1', type=str, default="20201005-20201122")  # 指定训练数据时间
    # ----------Para for function : list_remark
    parser.add_argument('--list_num', type=int, default=150)  # 指定清单输出基站个数

    # ----------Para for function : train_TF
    parser.add_argument('--input_len', type=int, default=7)  #  输入步长(天数)
    parser.add_argument('--target_long', type=int, default=3)  # 预测天数
    parser.add_argument('--lgb_lr', type=float, default=0.10)  # lgb模型参数
    parser.add_argument('--lgb_max_depth', type=int, default=5)
    parser.add_argument('--lgb_num_leaves', type=int, default=100)
    parser.add_argument('--lgb_num_boost_round', type=int, default=1)#迭代次数
    parser.add_argument('--xgb_lr', type=float, default=0.30)  # xgb模型参数
    parser.add_argument('--xgb_max_depth', type=int, default=5)
    parser.add_argument('--xgb_num_leaves', type=int, default=32)
    parser.add_argument('--xgb_n_estimators', type=int, default=1) #迭代次数
    parser.add_argument('--xgb_loss_func', type=str, default="original")  # xgb 的损失函数， original, focal_loss, weighted_loss
    parser.add_argument('--xgb_show_loss', type=bool, default=False)
    parser.add_argument('--focal_gamma', type=float, default=2)
    parser.add_argument('--imbalance_alpha', type=float, default=0.50)
    parser.add_argument('--drop_hour_duplicate', type=bool, default=False)
    parser.add_argument('--emb_size', type=int, default=50) # embedding 模型参数
    parser.add_argument('--emb_window', type=int, default=7)
    parser.add_argument('--emb_iter', type=int, default=1) #迭代次数
    parser.add_argument('--threshold', type=float, default=0.50)  # 分类阈值
    parser.add_argument('--model_weight', type=str, default="f1")  # 使用什么指标作为模型的权重, f1, recall, precision, rec_prec
    parser.add_argument('--use_weighted_avg', type=bool, default=False)  # 是否对模型加权
    parser.add_argument('--test_size', type=float, default=0.20)  # (交叉验证)测试数据集比例
    parser.add_argument('--ft_n_job', type=float, default = int((multiprocessing.cpu_count())))  # t退服多线程个数
    # parser.add_argument('--ft_n_job', type=float, default = -1)  # t退服多线程个数

    para = parser.parse_args()
    return para
