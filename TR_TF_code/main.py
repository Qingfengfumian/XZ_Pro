import sys
import datetime
from data_process import data_process_run
from source_data_multi import source_data_run
from run_train_1 import run_train_TR
from run_predict import run_predict_Pre
from effect_eva import run_effect_eva

def main(args):
    # 华为
    real_warns = ["[衍生告警]动环告警导致4G基站退服","小区不可用告警","网元连接中断","[衍生告警]同一基站发生所有LTE小区不可用告警","[衍生告警]同一基站发生多条LTE小区不可用告警"]  #4G华为
    # real_warns = ["NR小区不可用告警","gNodeB退服告警","[衍生告警]华为5G基站退服","网元连接中断","[衍生告警]动环告警导致5G基站退服","[衍生告警]同一基站发生多条NR小区不可用告警"] # 5G华为
    # 中兴
    # real_warns = ["[衍生告警]中兴部分LTE小区退服", "LTE小区退出服务","基站退出服务", "RRU链路断", "网元断链告警"]  # 4G中兴
    # real_warns = ["DU小区退服", "基站DU退服", "RRU链路断", "网元链路断", "[衍生关联]中兴部分NR小区退服", "[衍生告警]中兴5G基站退服"] # 5G中兴

    Org_path = '/home/XZ_Pro_AIOps/TR_TF_code/XZ_4G_HW/LS_4G_HW/'
    Org_path_pro = '/home/XZ_Pro_AIOps/TR_TF_code/XZ_4G_HW/LS_4G_HW/'
#     Org_path = '/root/LN_Pre/DL_4G_HW/'
#     Org_path_pro = '/root/LN-4G-HW/'
    Factory_C = 'HW'
    City_name = 'QS'
    City_name_pro = 'QS'
    mode_name = '4G'
    date_low_train = '20220208'
    date_high_train = '20220710'
    date_low_pre = '20220703'
    date_high_pre = '20220731'

    # 合并原始数据
    data_process_run(Org_path,Factory_C,mode_name,date_low_train,date_high_train,date_low_pre,date_high_pre)
    # 生成训练样本和测试样本
    source_data_run(Org_path, Factory_C, date_low_train, date_high_train, date_low_pre, date_high_pre,real_warns)
    # 模型训练
    run_train_TR(Org_path, Factory_C, City_name, mode_name, date_low_train, date_high_train,real_warns)
    # 模型测试
    run_predict_Pre(Org_path, Factory_C, City_name, mode_name, date_low_train, date_high_train, date_low_pre, date_high_pre,real_warns)
    # 模型结果验证
    run_effect_eva(Org_path, Factory_C, City_name, mode_name,Org_path_pro,City_name_pro)
    
if __name__ == "__main__":
    start_run_t = datetime.datetime.now()
    main(sys.argv)
    end_run_t = datetime.datetime.now()
    spend_t = round((end_run_t - start_run_t).seconds / 60, 1)
    print("任务结束，总耗时{}分钟。".format(str(spend_t)))
