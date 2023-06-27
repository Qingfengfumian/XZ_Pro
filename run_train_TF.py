import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import classification_report


from model import build_emb_model,build_xgb_model,build_lgb_model,load_model,evaluate_model_result
from utils import load_from_json,load_model_infos,find_new_file,save_model_infos,mkdir



def train_TF(para):
    input_path = "./Data/{}/Alert_Samp/Samp_{}/退服_样本_Xdata_{}.csv".format(para.distr, para.ftype,para.train_date1)
    label_path = "./Data/{}/Alert_Samp/Samp_{}/退服_样本_ydata_{}.csv".format(para.distr, para.ftype,para.train_date1)
    for path in [input_path, label_path]:
        if not os.path.exists(path):
            raise FileNotFoundError("data file '{}' does not exist, please check again or use the 'gen_train_data' mode first.".format(path))
    Xdata = pd.read_csv(input_path)
    ydata = pd.read_csv(label_path, header=None)
    xgbms = build_xgb_model(para, Xdata, ydata)
    lgbms = build_lgb_model(para, Xdata, ydata)

def evaluate_TF(para, data_pkg):
    # 执行预测
    pred_input = data_pkg[0]
    y_label = data_pkg[1]
    pred_output1 = np.zeros(shape=[pred_input.shape[0]])
    pred_output2 = np.zeros(shape=[pred_input.shape[0]])
    if para.use_weighted_avg:
        xgbm_weights = load_from_json(os.path.join(para.model_path, "lgb_model-weights.json"))
        lgbm_weights = load_from_json(os.path.join(para.model_path, "xgb_model-weights.json"))
    else:
        xgbm_weights = {ii:1 for ii in range(1,6)}
        lgbm_weights = {ii:1 for ii in range(1,6)}
    xgbm_weights = {str(k):v for k, v in xgbm_weights.items()}
    lgbm_weights = {str(k):v for k, v in lgbm_weights.items()}
    for i in range(1, 6):
        xgb_model_path = os.path.join(r"./Data/{}/Inter_Result/{}_退服_模型/".format(para.distr,para.ftype), "xgb_model_{}.pkl".format(i))
        lgb_model_path = os.path.join(r"./Data/{}/Inter_Result/{}_退服_模型/".format(para.distr,para.ftype), "lgb_model_{}.pkl".format(i))
        xgbmd = load_model(xgb_model_path)
        lgbmd = load_model(lgb_model_path)
        try:
            pred_output1 += xgbmd.predict_proba(pred_input, ntree_limit=xgbmd.best_iteration)[:, 1]*xgbm_weights[str(i)]
        except:
            pred_output1 += xgbmd.predict(xgb.DMatrix(pred_input, label=None), ntree_limit=xgbmd.best_iteration) * xgbm_weights[str(i)]
        pred_output2 += lgbmd.predict(pred_input, num_iteration=lgbmd.best_iteration)*lgbm_weights[str(i)]
    xgb_weight_sum = np.sum([value for value in xgbm_weights.values()])
    lgb_weight_sum = np.sum([value for value in lgbm_weights.values()])
    pred_output_prob = (pred_output1 / xgb_weight_sum + pred_output2 / lgb_weight_sum) / 2
    pred_output_label = [int(v >= para.threshold) for v in pred_output_prob]
    eval_result1 = classification_report(y_label, pred_output_label, output_dict=True)
    eval_result2 = evaluate_model_result(y_label, pred_output_label)
    print("模型验证报告:\n", eval_result1)
    print(eval_result2)
    result_path = r"./Data/{}/Inter_Result/{}_退服_模型/{}_模型验证报告.txt".format(para.distr,para.ftype,para.train_date1)
    with open(result_path, "w") as txt_file:
        txt_file.write("模型验证报告:\n{},\n\n{}.".format(eval_result1, eval_result2))