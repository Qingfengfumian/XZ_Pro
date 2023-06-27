import sys
import datetime
from utils import params_setup
from data_deal import distr_del_cat
# from run_train_XJ import train_XJ
# from run_train_TF import train_TF
from run_predict import predict,list_remark,validation
from data_proc_XJ_4G import proc_XJ_4G
from data_proc_XJ_5G import proc_XJ_5G
from data_proc_TF_4G import proc_TF_4G
from data_proc_TF2_4G import proc_TF_4G as proc_TF2_4G
from data_proc_TF2_5G import proc_TF_5G as proc_TF2_5G
from run_predict_TF import predict_TF
from run_predict_XJ_4G import predict_XJ_4G
from run_predict_XJ_5G import predict_XJ_5G
from run_predict_TF2_4G import predict_TF_4G as predict_TF2_4G
from run_predict_TF2_5G import predict_TF_5G as predict_TF2_5G
import pandas as pd

def main(args):
    date_low = '20210504'
    date_high = '20210505'
    # date_range = pd.date_range(date_low, date_high)
    # for i in date_range:
    for i in [1]:
        # date_fanwei = i + datetime.timedelta(days=6)
        # date_time = str(i).split(' ')[0].replace('-', '') + '-' + str(date_fanwei).split(' ')[0].replace('-', '')
        para = params_setup()
        # para.date = date_time
        # run_create_dir(para)# 生成data初始目录
        distr_del_cat(para)  # 处理告警 判断是否已处理，已处理跳过
        if para.mode == 'train_XJ':
            proc_XJ_4G(para)  # 生成样本 train模式对区间已处理文件合并
            # train_XJ(para)
        if para.mode == 'train_TF':
            proc_TF_4G(para)  #生成样本 train模式合并文件处理
            # train_TF(para)
        if para.mode == 'predict':
            proc_XJ_4G(para)
            predict_XJ_4G(para)
            proc_XJ_5G(para)
            predict_XJ_5G(para)
            for distr in para.distr_list:
                if distr in ['AA']:
                    pred_ft_data, ft_cols = proc_TF_4G(para,distr)
                    predict_TF(para,pred_ft_data, ft_cols,distr)
                elif distr in ['铜川','汉中','宝鸡','咸阳','西安','商洛','安康', '渭南', '延安', '榆林']:
                    pred_ft_data6, ft_cols6 = proc_TF2_4G(para, distr, day=6)
                    pred_ft_data7, ft_cols7 = proc_TF2_4G(para, distr, day=7)
                    predict_TF2_4G(para, pred_ft_data6, ft_cols6, pred_ft_data7, ft_cols7, distr)
                    pred_ft_data6_5, ft_cols6_5 = proc_TF2_5G(para, distr, day=6)
                    pred_ft_data7_5, ft_cols7_5 = proc_TF2_5G(para, distr, day=7)
                    predict_TF2_5G(para, pred_ft_data6_5, ft_cols6_5, pred_ft_data7_5, ft_cols7_5, distr)
            predict(para)
            list_remark(para)
            validation(para)

if __name__ == "__main__":
    start_run_t = datetime.datetime.now()
    main(sys.argv)
    end_run_t = datetime.datetime.now()
    spend_t = round((end_run_t - start_run_t).seconds / 60, 1)
    print("任务结束，总耗时{}分钟。".format(str(spend_t)))
