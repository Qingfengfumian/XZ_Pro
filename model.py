# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from xgboost import XGBClassifier
import xgboost as xgb
from loss_func import Weight_Binary_Cross_Entropy, Focal_Binary_Loss, evalerror
import lightgbm as lgb
from gensim.models import word2vec
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
import os
import joblib
import logging
import copy
import json
from utils import mkdir, real_warns_4G, save_to_json

# 训练word2vec告警类型编码模型
def build_emb_model(para, warn_series, all_warns0):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    wv_model = word2vec.Word2Vec(warn_series, size=para.emb_size, window=para.emb_window, min_count=1,
                                 iter=para.emb_iter)
    emb_path = './Data/{}/Inter_Result/{}_退服_模型/'.format(para.distr, para.ftype)
    if os.path.exists(emb_path):
        mkdir(emb_path)
    wv_model.save("./Data/{}/Inter_Result/{}_退服_模型/embedding_model_{}.pkl".format(para.distr, para.ftype, para.train_date1))
    # 计算每种告警类型的得分
    warn_scores = {}
    for warn_name in all_warns0:
        sim_score = 0
        if warn_name in wv_model.wv.vocab.keys():
            for real_warn in real_warns_4G:
                sim_score += wv_model.wv.similarity(real_warn, warn_name)
        else:
            print(warn_name)
        warn_scores[warn_name] = sim_score if sim_score > 0 else 0  # 最小值设为0

    max_score = max(warn_scores.values())
    warn_scores2 = copy.deepcopy(warn_scores)

    # 重置退服告警的编码
    for real_warn in real_warns_4G:
        warn_scores2[real_warn] = 2 * max_score
    # 保存编码结果
    save_path = r"./Data/{}/Inter_Result/{}_特征编码/".format(para.distr,para.ftype)
    if os.path.exists(save_path)==False:
        mkdir(save_path)
    with open(os.path.join(save_path,"{}_labelencoder.json".format(para.train_date1)), "w") as json_f:  # 保存到数据目录（不同模型可能共用数据）
        json.dump(warn_scores2, json_f)
    return warn_scores2

def build_xgb_model(para, Xdata, ydata, eval_data=None):
    # todo: select different loss funcs; to validate each model on test data to get the f1-score.
    # 采用5折交叉验证
    folds = GroupKFold(n_splits=5)
    groups = Xdata['week']

    x_cols = [col for col in Xdata.columns if not col =="week"]
    Xdata = Xdata[x_cols]

    def recall_add_precision(true_label, pred_label):
        return recall_score(true_label, pred_label) + precision_score(true_label, pred_label)
    indicators_dict = {"f1":f1_score, "recall":recall_score, "precision":precision_score, "rec_prec":recall_add_precision}
    indicator = indicators_dict[para.model_weight]

    # img_path = os.path.join(img_dir, para.model_id)
    # mkdir(img_path)

    xgbms = []
    model_weights = {}
    for n_fold, (train_index, test_index) in enumerate(folds.split(Xdata, ydata, groups), start=1):
        print('the {}-th cross-validation train'.format(n_fold))
        train_x, train_y = (Xdata.iloc[train_index], ydata.iloc[train_index])
        test_x, test_y = Xdata.iloc[test_index], ydata.iloc[test_index]
        if para.xgb_loss_func == "original":
            sub_xgbm = XGBClassifier(max_depth=para.xgb_max_depth, learning_rate=para.xgb_lr,
                                 n_estimators=para.xgb_n_estimators,
                                 objective="binary:logistic",
                                 nthread=-1, gamma=0, min_child_weight=1,
                                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                 base_score=0.5, seed=0, missing=None)
            train_dt = (train_x, train_y)
            eval_dt = (test_x, test_y)
            sub_xgbm = sub_xgbm.fit(Xdata, ydata, eval_set=[train_dt, eval_dt], eval_metric=['logloss','auc','error'],
                                early_stopping_rounds=50, verbose=True)
            pred_y = sub_xgbm.predict_proba(test_x, ntree_limit=sub_xgbm.best_iteration)[:, 1]
        else:  # "focal_loss" or "weighted_loss"
            para_dict = {'max_depth': para.xgb_max_depth,
                         'eta': para.xgb_lr,  # 目前最优参数
                         'silent': False,
                         'objective': 'binary:logitraw',
                         'eval_metric': ['logloss', 'auc', 'error'],  # "auc",# 'logloss',
                         'booster': 'gbtree'}
            num_round = 500
            dtrain = xgb.DMatrix(train_x, label=train_y)  # weight=weights
            dvalid = xgb.DMatrix(test_x, label=test_y)
            eval_list = [(dtrain, 'train'), (dvalid, 'validate')]
            if para.xgb_loss_func == "focal_loss":
                loss_obj = Focal_Binary_Loss(gamma_indct=para.focal_gamma).focal_binary_object  # 0.10 最佳参数
            else:  # para.xgb_loss_func == "weighted_loss":
                loss_obj = Weight_Binary_Cross_Entropy(imbalance_alpha=para.imbalance_alpha).weighted_binary_cross_entropy  # 0.50
            # fit the classfifier
            sub_xgbm = xgb.train(para_dict, dtrain, num_round, eval_list, obj=loss_obj, feval=evalerror,early_stopping_rounds=50, verbose_eval=10)
            pred_y = sub_xgbm.predict(xgb.DMatrix(test_x,label=None), ntree_limit=sub_xgbm.best_iteration)
        subm_weight = get_model_weight(indicator, test_y, pred_y, 0.40)  # para.threshold, 0.50(?)
        model_weights[n_fold] = subm_weight
        save_path = r"./Data/{}/Inter_Result/{}_退服_模型/".format(para.distr,para.ftype)
        model_path_i = os.path.join(save_path, "xgb_model_{}_{}.pkl".format(n_fold,para.train_date1))
        joblib.dump(sub_xgbm, model_path_i)  # 保存
        xgbms.append(sub_xgbm)

        if para.xgb_show_loss == True:
            results = sub_xgbm.evals_result()
            epochs = len(results['validation_0']['error'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = pyplot.subplots(211)
            pyplot.plot(x_axis, results['validation_0']['logloss'], label='Train')
            pyplot.plot(x_axis, results['validation_1']['logloss'], label='Test')  #?
            pyplot.legend()
            pyplot.ylabel('Log Loss')
            pyplot.title('XGBoost Log Loss')
            # pyplot.show()
            # pyplot.savefig(os.path.join(img_path, "xgb_log_loss_{}".format(str(n_fold))))
            # plot classification error
            fig2, ax2 = pyplot.subplots(212)
            pyplot.plot(x_axis, results['validation_0']['error'], label='Train')
            pyplot.plot(x_axis, results['validation_1']['error'], label='Test')  #?
            pyplot.legend()
            pyplot.ylabel('Classification Error')
            pyplot.title('XGBoost Classification Error')
            # pyplot.show()
            # pyplot.savefig(os.path.join(img_path, "xgb_train_loss_{}.png".format(str(n_fold))))  # 保存图片

    # save_to_json(model_weights, os.path.join(para.model_path, "xgb_model-weights.json"))
    print("xgboost 模型训练完毕, 已更新在: {}。".format(save_path))
    return xgbms

def build_lgb_model(para, Xdata, ydata): # todo: select different loss funcs; to validate each model on test data to get the f1-score.
    param = {
        'num_leaves': para.lgb_num_leaves,
        'objective': 'binary',
        'max_depth': para.lgb_max_depth,
        'learning_rate': para.lgb_lr,
        'max_bin': 300,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'min_data_in_leaf': 28,
        'verbose': -1}
    param['metric'] = ['auc', 'binary_logloss']

    # 采用5折交叉验证
    folds = GroupKFold(n_splits=5)
    groups = Xdata['week']

    x_cols = [col for col in Xdata.columns if not col in ["week"]]
    Xdata = Xdata[x_cols]

    def recall_add_precision(true_label, pred_label):
        return recall_score(true_label, pred_label) + precision_score(true_label, pred_label)
    indicators_dict = {"f1":f1_score, "recall":recall_score, "precision":precision_score, "rec_prec":recall_add_precision}
    indicator = indicators_dict[para.model_weight]

    lgbms = []
    model_weights = {}
    for n_fold, (train_index, test_index) in enumerate(folds.split(Xdata, ydata, groups),start=1):
        lgb_train_data = lgb.Dataset(Xdata.iloc[train_index], ydata.iloc[train_index])
        test_x, test_y = Xdata.iloc[test_index], ydata.iloc[test_index]
        lgb_valid_data = lgb.Dataset(test_x, test_y)
        sub_lgbm = lgb.train(params=param,train_set=lgb_train_data, valid_sets=lgb_valid_data, valid_names="validation",
                           num_boost_round=para.lgb_num_boost_round, early_stopping_rounds=50, verbose_eval=10)
        pred_y = sub_lgbm.predict(test_x, num_iteration=sub_lgbm.best_iteration)
        subm_weight = get_model_weight(indicator, test_y, pred_y, 0.50)  # para.threshold
        model_weights[n_fold] = subm_weight
        save_path = r"./Data/{}/Inter_Result/{}_退服_模型/".format(para.distr, para.ftype)
        model_path_i = os.path.join(save_path, "lgb_model_{}_{}.pkl".format(n_fold,para.train_date1))
        joblib.dump(sub_lgbm, model_path_i) # 保存
        lgbms.append(sub_lgbm)

    # save_to_json(model_weights, os.path.join(para.model_path, "lgb_model-weights.json"))
    print("lightgbm 模型训练完毕, 已更新在: {}。".format(save_path))
    return lgbms

# 模型评估函数
def evaluate_model_result(true_value, pred_values):
    f1 = f1_score(true_value, pred_values)
    acc = accuracy_score(true_value, pred_values)
    prec = precision_score(true_value, pred_values)
    rec = recall_score(true_value, pred_values)
    cf_mat = confusion_matrix(true_value, pred_values)
    return {"f1:":f1, "acc:":acc, "prec:":prec, "rec:":rec, "confusion:":cf_mat.tolist()}


# 获取模型权重
def get_model_weight(indicator, true_value, raw_ouput, threshold):
    pred_label = [int(v>=threshold) for v in raw_ouput]
    return indicator(true_value, pred_label)


def load_model(model_file):
    if not os.path.exists(model_file):
        raise FileNotFoundError("model file '{}' does not exist, please check again.".format(model_file))
    local_model = joblib.load(model_file)
    return local_model