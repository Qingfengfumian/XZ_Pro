import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model,Sequential,load_model
import time
from keras.layers import Dense, Input,Dropout,Lambda
from keras.optimizers import Adam,RMSprop
from keras import losses
from keras import initializers
import os
import pickle

from utils import find_new_file,find_new_model_file,mkdir

def data_loader(distr, date, ftype):
   
    Test_samp = pd.read_csv(r'./Data/{}/Alert_Samp/Samp_predict_{}/故障_样本_{}_4G.csv'
                                 .format(distr,ftype, date), encoding='gbk', index_col=0,engine='python') 
    x_test = Test_samp.values[:, :]
    # 标准化
    scaled = np.copy(x_test)
    ss_filepath = find_new_file( "./Data/{}/Inter_Result/4G/{}_故障_标准化/".format(distr,ftype))
    with open(ss_filepath, 'rb') as f:
        ss = pickle.load(f)
        f.close()
    x_pred = ss.transform(scaled)
    return x_pred, Test_samp
def ae_predictor(distr, x_pred, ftype):
    def mean_squa(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    start = time.clock()
    model_filepath = find_new_model_file(
        r"./Data/{}/Inter_Result/4G/{}_故障_模型/".format(distr, ftype), modtype='模型AE')
    model = load_model(r"./Data/{}/Inter_Result/4G/{}_故障_模型/{}".format(distr, ftype, model_filepath),custom_objects={'mean_squa': mean_squa})
    decoded_ae = model.predict(x_pred)
    print('预测完成 in %s seconds' % (time.clock() - start))
    return decoded_ae
def vae_predictor(distr, x_pred, ftype):
    original_dim = len(x_pred[0])  # 173
    latent_dim = 16  # 隐变量取2维只是为了方便后面画图
    intermediate_dim_1 = round(original_dim / 2)
    intermediate_dim_2 = round(original_dim / 4)
    inputs = Input(shape=(original_dim,))
    h1 = Dense(intermediate_dim_1, activation='relu', kernel_initializer=initializers.lecun_normal(seed=None))(
        inputs)
    h2 = Dense(intermediate_dim_2, activation='relu', kernel_initializer=initializers.lecun_normal(seed=None))(h1)
    # 算p(Z|X)的均值和方差
    z_mean = Dense(latent_dim)(h2)
    #    epsilon = K.random_no(h2)
    z_log_var = Dense(latent_dim)(h2)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon_std = 1.0
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0.0,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    decoder_h1 = Dense(intermediate_dim_2, activation='relu')(z)
    decoder_h2 = Dense(intermediate_dim_1, activation='relu')(decoder_h1)
    outputs = Dense(original_dim, activation='selu')(decoder_h2)
    vae = Model(inputs, outputs)

    def vae_loss(inputs, outputs):
        reconstruction_loss = losses.mse(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    adam = Adam(lr=0.0001)
    vae.compile(optimizer=adam, loss=vae_loss)

    start = time.clock()
    model_filepath = find_new_model_file(r"./Data/{}/Inter_Result/4G/{}_故障_模型/".format(distr,ftype), modtype='模型VAE')
    vae = load_model(r"./Data/{}/Inter_Result/4G/{}_故障_模型/{}".format(distr,ftype, model_filepath),
                     custom_objects={'vae_loss': vae_loss})
    decoded_vae = vae.predict(x_pred)
    print('预测完成 in %s seconds' % (time.clock() - start))
    return decoded_vae
def err_cal(x_pred, decoded_data, test_samp, ftype,distr):
    # --------------------------------------- 在数据后增加一列 误差和 ---------------------------------------
    Diff = abs(x_pred - decoded_data)
    Test_samp = test_samp.copy()
    Diff_val_init = pd.DataFrame(Diff, columns=list(Test_samp.columns.values), index=list(Test_samp.index.values))
    Diff_val_df = Diff_val_init
#     Diff_val_init.to_csv(
#         './1Diff_val_init.csv',
#         encoding='gbk')
    print('开始-零次告警置零、特殊异常处理x2')
    start = time.clock()
    Test_samp_1 = Test_samp
    Test_samp_1[Test_samp_1 != 0] = 1  #！=0 置1
    Diff_val_df = Test_samp_1 * Diff_val_df  #零次告警置零
    if ftype == '中兴':
        Test_samp_2 = Test_samp[[  # 中兴
            '光模块接收光功率异常', '输入电压异常']]
    elif ftype == '华为':
        Test_samp_2 = Test_samp[[  # 华为
            '网元连接中断'
            ]]# '电源模块异常告警','市电输入异常告警','基站直流供电异常告警','单板输入电压异常告警','CXU输入电源能力不足告警
    else:
        Test_samp_2 = Test_samp
        print('厂家类型错误！')
    is_zero = Test_samp_2.apply(lambda x: x.sum(), axis=1) #是否出现特殊告警
    is_zero = is_zero.map(lambda x: 0 if x==1 else 0) #is_zero==0 这一步跳过
    BZ_dict1 = is_zero.copy()
    BZ_dict1[BZ_dict1 != 0] = 1
    BZ_dict1[BZ_dict1 == 0] = 0
    Diff_val_df['BZ_dict1'] = BZ_dict1
    Diff_val_df['BZ_dict1'] = Diff_val_df['BZ_dict1'].map(lambda x: str(x))
    is_zero[is_zero != 0] = 2
    is_zero[is_zero == 0] = 1
    Diff_val_df.iloc[:, :-1] = Diff_val_df.iloc[:, :-1].mul(is_zero, axis=0)
    # 退服类告警增加不同权重
    Alert_weight = pd.read_excel(r"./Data/汇总/Project_Opt/华为_告警加权.xlsx")
    Alert_weight1 = Alert_weight[['告警标题','权重']]
    for idx in range(len(Alert_weight1['告警标题'])):
        try:
            Diff_val_df[Alert_weight1.iloc[idx,0]] = Diff_val_df[Alert_weight1.iloc[idx,0]]*Alert_weight1.iloc[idx,1]
        except:
            continue
    print('完成-零次告警置零、特殊异常处理x2 in {} seconds'.format(time.clock() - start))

#     Diff_val_df.to_csv(
#         './Diff_val_hwweight.csv',
#         encoding='gbk')
    Diff_val_df['enbid'] = Test_samp.index.astype('str').map(lambda x: x.split('|')[0]).tolist()
    import datetime
    try:
        Diff_val_df['date'] = Test_samp.index.astype('str').map(
            lambda x: datetime.datetime.strptime(x.split('|')[1], '%Y-%m-%d').date()).tolist()
    except:
        Diff_val_df['date'] = Test_samp.index.astype('str').map(
            lambda x: datetime.datetime.strptime(x.split('|')[1], '%Y/%m/%d').date()).tolist()
    Diff_val_df['enbid+date'] = Diff_val_df.index
    Diff_val_df1 = Diff_val_df.copy()
    # ---------新方法：按星期加权  in 18.030155899999997 seconds--------
    def mynew(Diff_val_df):
        print('开始-按日期加权')
        import time
        start = time.clock()
        map_dict = {
            '0': 1,
            '1': 0.9,
            '2': 0.8,
            '3': 0.5,
            '4': 0.3,
            '5': 0.2,
            '6': 0.1
        }
        Max_date = Diff_val_df['date'].max()
        Diff_val_df['date_stamp'] = Diff_val_df['date'].map(lambda x: str((Max_date - x).days))
        Diff_val_df['date_weight'] = Diff_val_df['date'].map(lambda x: map_dict[str((Max_date - x).days)])
        Diff_val_df.iloc[:, :-6] = Diff_val_df.iloc[:, :-6].mul(Diff_val_df['date_weight'], axis=0)

#         Diff_val_df.to_csv(
#         './Diff_val_weight.csv',
#         encoding='gbk')

        Diff_val_df1 = Diff_val_df.drop(columns=['date', 'enbid+date', 'date_weight'],axis=1)
        Diff_val_df1['BZ_dict1'] = Diff_val_df1['BZ_dict1'].map(lambda x: int(x))
        Diff_val_df3 = Diff_val_df1.groupby('enbid')['enbid'].count()
        df = {'enbid': Diff_val_df3.index, 'days_cnt': Diff_val_df3.values}
        Diff_val_df4 = pd.DataFrame(df)
        from joblib import Parallel, delayed
        import multiprocessing
        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
            return pd.concat(retLst, axis=0)
        import time
        start = time.clock()
        def week_sum(df):
            col = 'enbid'
            label = df[col].tolist()[0]
            weight_sort = sorted(df['date_stamp'].tolist())
            weight_str = [str(i) for i in weight_sort]
            df.drop([col, 'date_stamp'], axis=1, inplace=True)
            df = df.apply(lambda x: x.sum(), axis=0)
            df['告警时间戳'] = ','.join(weight_str)
            df[col] = label
            return df
        multi_res = applyParallel(Diff_val_df1.groupby('enbid'), week_sum)
        n = len(set(multi_res.index.tolist()))
        len0 = int(len(multi_res.index.tolist()) / n)
        col0 = multi_res.index.tolist()[0:n]
        nd = np.array(multi_res.values)
        DF = pd.DataFrame(nd.reshape(len0, n), columns=col0)
        Diff_val_df2 = DF
        Diff_val_df2['enbid'] = Diff_val_df2['enbid'].map(lambda x: str(x).split('.')[0])
        Diff_val_df5 = pd.merge(Diff_val_df2, Diff_val_df4, on='enbid')
        Diff_val_df5['err_sum'] = Diff_val_df5.iloc[:, 0:-4].sum(axis=1)
        Diff_val_df5['BZ_dict1'] = Diff_val_df5['BZ_dict1'].map(lambda x: str(x).split('.')[0])
        Diff_val_df5['days_cnt'] = Diff_val_df5['days_cnt'].map(lambda x: str(x))

        Diff_val_df5['备注'] = Diff_val_df5['BZ_dict1'] + ',' + Diff_val_df5['days_cnt']
        Diff_val_df5.drop(['BZ_dict1', 'days_cnt'], axis=1, inplace=True)
        Diff_val_df5['备注项个数'] = Diff_val_df5['备注'].map(
            lambda x: len(x.replace('0', '').replace(',', '').replace(' ', '')))
        Diff_val_df6 = Diff_val_df5.sort_values(by='err_sum', axis=0, ascending=False, inplace=False)
        Diff_val_df6['level'] = range(len(Diff_val_df6))
#         Diff_val_df6.to_csv(
#         './Diff_val_df6.csv',
#         encoding='gbk')
        print('完成-按日期加权 in {} seconds'.format(time.clock() - start))
        return Diff_val_df6  # 增加了备注项['x','y'] x='1'有典型隐性故障 y='3'三天出现告警 和备注项个数
    res = mynew(Diff_val_df1)
    return res
def GongCan_process(gongcanfile):
    try:
        GongCan = pd.read_csv(gongcanfile, encoding='gbk', engine='python')
    except:
        GongCan = pd.read_csv(gongcanfile, encoding='utf-8', engine='python')
    GongCan_W = GongCan[['ENODEB_ID', '基站名称']]
    GongCan_W['ENODEB_ID'] = GongCan_W['ENODEB_ID'].map(lambda x:str(x).split("-")[0].split('.')[0])
    GongCan_W_del = GongCan_W.drop_duplicates(subset=['ENODEB_ID', '基站名称'], keep='first')
    GongCan_W_del = GongCan_W_del.rename(columns={'ENODEB_ID': 'enbid', '基站名称': '所属基站'})
    return GongCan_W_del
def JF_concat(distr,data,test_samp):
    # 机房合并,增加异常列
    ZX_Samp_Sum_sort2 = data
    GongCan = GongCan_process(r"./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv")
    ZX_Samp_Sum_sort2['enbid'] = ZX_Samp_Sum_sort2['enbid'].astype(str)
    ZX_Samp_Sum_merge = ZX_Samp_Sum_sort2

    # ZX_Samp_Sum_merge.to_csv(
    #     './{}_{}_4Samp_add_JF.csv'.format(date, ftype),
    #     encoding='gbk', index=None)
    # ZX_Samp_Sum_merge = pd.read_csv('./{}_{}_4Samp_add_JF.csv'.format(date, ftype),encoding='gbk')
    ##————————————统计异常告警发生次数
    Test_samp = test_samp
    Test_samp.reset_index(drop=False, inplace=True)
    Test_samp['enbid+date'] = Test_samp['enbid+date'].apply(lambda x: x.split('|')[0])
    Test_samp = Test_samp.groupby(by=['enbid+date']).sum()

    Diff_threshold = 0.99
    Diff_row_max = 100  # 取前8个误差最大的维度


    print('开始-备注项机房合并')
    start = time.clock()

    def gener_BZ3(df):
        maxdays = df['备注'].map(lambda x: int(x.split(',')[1])).max()
        ty_alert = '0' if df['备注项个数'].max() == 1 else '1'
        df['备注'] = ty_alert + ',' + str(maxdays) + ',' + str(df.shape[0])
        del df['备注项个数']
        return df

    ZX_Samp_Sum_merge = ZX_Samp_Sum_merge.groupby(['enbid']).apply(gener_BZ3)
    print('完成-备注项机房合并 in {} seconds'.format(time.clock() - start))
    # print('开始-所属机房误差合并')
    # start = time.clock()
    # Samp_Sum_merge = ZX_Samp_Sum_merge.groupby('所属机房').apply(lambda x: x.iloc[:, :-4].sum())
    # Samp_Sum_merge.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='所属机房')
    # Samp_Sum_merge.insert(Samp_Sum_merge.shape[1] - 1, '所属机房', Samp_Sum_merge.pop('所属机房'))
    # Samp_Sum_merge_all = pd.merge(Samp_Sum_merge, ZX_Samp_Sum_merge[['备注', '所属机房']])
    # Samp_Sum_merge_all.drop_duplicates(subset='所属机房', keep='first', inplace=True)
    # print('完成-所属机房误差合并 in {} seconds'.format(time.clock() - start))
    Samp_Sum_merge_all = ZX_Samp_Sum_merge
    maxval = Samp_Sum_merge_all['err_sum'].max()
    Samp_Sum_merge_all['err_sum_score'] = Samp_Sum_merge_all['err_sum'].apply(lambda x: x / maxval)
    Samp_Sum_merge_all.reset_index(drop=True, inplace=True)
    Samp_Sum_merge_all = Samp_Sum_merge_all.drop(columns=['level'])

    print('开始-增加具体异常列')
    start = time.clock()
    Unnormal = []
    unnormal_alarm_freq = []
    sum_error = Samp_Sum_merge_all['err_sum']
    for i in range(Samp_Sum_merge_all.shape[0]):
        Diff_zh_row = Samp_Sum_merge_all.iloc[i, :-5]  # 去除sum_new enbid 备注 告警时间戳 err_sum_score 后几列
        enbid = int(Samp_Sum_merge_all.loc[i]['enbid'])
        Diff_zh_row_sort = Diff_zh_row.argsort()
        linshia = Diff_zh_row_sort.tolist()[::-1]
        sum_val = 0
        error = sum_error[i] * Diff_threshold
        if error == 0:
            nanstr = []
            Unnormal.append(nanstr)
            unnormal_alarm_freq.append(nanstr)
        else:
            for j in range(len(linshia)):
                try:
                    sum_val += Diff_zh_row[linshia[j]]
                except:
                    print(j, sum_val)
                if ((sum_val >= error) or (j >= Diff_row_max)):
                    break
            linshib = linshia[:j + 1:1]  # 返回数据从小到大的索引值
            for k in linshib:
                if (Diff_zh_row[k] == 0):
                    linshib.remove(k)
            linshid = list(Samp_Sum_merge_all.columns[linshib])  # 返回告警中文名

            linshil = []
            for alertname in linshid:
                try:
                    linshid_sum = Test_samp.loc[str(enbid)][alertname]
                except:
                    linshid_sum = Test_samp.loc[int(enbid)][alertname]
                linshil.append(alertname + '：' + str(linshid_sum) + '次')
            if sum_error[i] == 0:
                print(i)
            linshie = list(Samp_Sum_merge_all.iloc[i, linshib] * 100 / sum_error[i])  # 返回误差值
            linshif = [('(' + str(linshib[i]) + ')' + linshid[i] + '(' + "{:.2f}".format(linshie[i]) + '%' + ')') for i
                       in
                       range(0, len(linshid))]
            Unnormal.append(str(linshif)[1:-1])  # 在dataframe最后插入告警及异常差值
            unnormal_alarm_freq.append(str(', '.join(linshil)))
    Samp_Sum_merge_all['巡检重点关注项目'] = Unnormal
    Samp_Sum_merge_all['重点关注影响业务告警发生频次'] = unnormal_alarm_freq
    print('完成-增加具体异常列 in {} seconds'.format(time.clock() - start))
    print('开始-巡检清单生成')
    start = time.clock()
    Samp_merge_drop2 = Samp_Sum_merge_all.dropna()

    def agg(bz):
        res = ''
        if (bz.split(',')[0] == '1'):
            res += '典型隐性故障;'
        if (int(bz.split(',')[1]) > 1):
            res += '连续多天异常(%s);' % (bz.split(',')[1])
        if (int(bz.split(',')[-1]) > 1):
            res += '单机房多站异常(%s)' % (bz.split(',')[-1])
        return res

    Samp_merge_drop2['备注'] = Samp_merge_drop2['备注'].map(lambda x: str(x)).apply(agg)
    Samp_sort = Samp_merge_drop2.sort_values(by='err_sum_score', ascending=False)
    Samp_sort['基站健康度异常程度'] = sum_error
    Samp_sort['巡检优先级'] = range(len(Samp_sort))
    Samp_sort.reset_index(inplace=True, drop=True)
    # Samp_sort.to_csv(
    #     orgpath + '/AIOps/Inter_Result/{}/{}_{}_5Samp_sort_{}.csv'.format(date, ftype, alertclass, model_type),
    #     encoding='gbk', index=None)

    Samp_sort1 = pd.merge(Samp_sort, GongCan,on='enbid',how='right')


    Samp_sort1 = Samp_sort1[['基站健康度异常程度', '巡检重点关注项目',  '重点关注影响业务告警发生频次','所属基站','enbid', '备注', '巡检优先级','告警时间戳']]

    Samp_sort1.reset_index(inplace=True, drop=True)

    print('完成-巡检清单生成 in {} seconds'.format(time.clock() - start))
    return Samp_sort1
def predict_XJ_4G(para):
    distr_list = para.distr_list
    date = para.date
    ftype = para.ftype
    for distr in distr_list:
        pred_data, Test_samp = data_loader(distr, date, ftype)

        decoded_ae = ae_predictor(distr, pred_data, ftype)
        decoded_vae = vae_predictor(distr, pred_data, ftype)
        Diff_val_process_ae = err_cal(pred_data, decoded_ae, Test_samp, ftype,distr)
        Diff_val_process_vae = err_cal(pred_data, decoded_vae, Test_samp, ftype,distr)
        start = time.clock()

        result_ae = JF_concat(distr,Diff_val_process_ae,Test_samp)
        print('完成-AE巡检清单生成 in {} seconds'.format(time.clock() - start))
        start = time.clock()
        result_vae = JF_concat(distr, Diff_val_process_vae, Test_samp)

        print('完成-VAE巡检清单生成 in {} seconds'.format(time.clock() - start))
        print('开始-AE+VAE结果合并')
        start = time.clock()
        result_ae.reset_index(drop=True, inplace=True)
        result_vae.reset_index(drop=True, inplace=True)

        Result_concat = pd.concat([result_ae, result_vae], axis=0, ignore_index=True)
        Result_sort = Result_concat.sort_values(by='巡检优先级', ascending=True)

        Result_sort['init_sort'] = Result_sort.index.values.tolist()
        def reply(x):
            if x >= (len(Result_sort['init_sort']) / 2):
                x = 'VAE'
            else:
                x = 'AE'
            return x

        Result_sort['推断模型'] = Result_sort['init_sort'].map(reply)

        Result_sort.drop(columns=['巡检优先级', 'init_sort'], inplace=True)

        Result_drop = Result_sort.drop_duplicates(subset='所属基站', keep='first')
        level = [i + 1 for i in range(Result_drop.shape[0])]
        Result_drop['巡检优先级'] = level

        print('完成-AE+VAE结果合并 in %s' % (time.clock() - start))
        # 增加典型隐形告警名--暂未使用
        if ftype == '中兴1':
            print('开始-增加典型隐形告警名（中兴）')
            start = time.clock()
            enbid = []
            Test_samp1 = Test_samp
            for enb in Test_samp1.index.values.tolist():
                enbid.append(str(enb).split('|')[0])
            Test_samp1['enbid'] = enbid
            GongCan = GongCan_process("./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv")
            Test_samp_merge = pd.merge(Test_samp1, GongCan)

            def jifang_cnt(df):
                for col in list(Test_samp_merge_drop.columns.values[:-1]):  # 所属机房
                    df[col] = df[col].sum()
                return df

            Test_samp_merge_drop = Test_samp_merge.drop(columns=['enbid'])
            Test_samp_merge = Test_samp_merge_drop.groupby('所属基站').apply(jifang_cnt)
            Test_samp_merge.drop_duplicates(subset=['所属基站'], keep='first', inplace=True)
            Inter = pd.merge(Test_samp_merge, Result_drop, how='left')
            Inter = Inter.dropna(subset=['巡检重点关注项目'])
            Inter1 = Inter[['光模块接收光功率异常', '输入电压异常', '巡检重点关注项目']]
            add_cont = []
            columns_name = Test_samp1.columns.values.tolist()
            for idx in Inter1.index.values.tolist():
                st = ''
                if Inter1.loc[idx, '光模块接收光功率异常'] > 0 and '光模块接收光功率异常' not in Inter1.loc[idx, '巡检重点关注项目']:
                    st += "'({})光模块接收光功率异常',".format(columns_name.index('光模块接收光功率异常'))
                if Inter1.loc[idx, '输入电压异常'] > 0 and '输入电压异常' not in Inter1.loc[idx, '巡检重点关注项目']:
                    st += "'({})输入电压异常',".format(columns_name.index('输入电压异常'))
                add_cont.append(st)
            Inter1['add_cont'] = add_cont
            print('完成-增加典型隐形告警名 in %s seconds' % (time.clock() - start))
            Inter['巡检重点关注项目'] = Inter['巡检重点关注项目'] + ',' + Inter1['add_cont']
            Inter['巡检重点关注项目'].map(lambda x: str(x).replace(';', ','))
            Fin_result = Inter[['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '所属基站', '备注', '推断模型', '告警时间戳']]
            Fin_result = Fin_result.sort_values(by='巡检优先级', ascending=True)
        elif ftype == '华为1':
            print('开始-增加典型隐形告警名（华为）')
            start = time.clock()
            enbid = []
            Test_samp1 = Test_samp
            for enb in Test_samp1.index.values.tolist():
                enbid.append(str(enb).split('|')[0])
            Test_samp1['enbid'] = enbid
            GongCan = GongCan_process(r"./Data/汇总/Project_Opt/GC-CELL-BS-ROOM.csv")
            Test_samp_merge = pd.merge(Test_samp1, GongCan)

            def jifang_cnt(df):
                for col in list(Test_samp_merge_drop.columns.values[:-1]):  # 所属机房
                    df[col] = df[col].sum()
                return df

            Test_samp_merge_drop = Test_samp_merge.drop(columns=['enbid'])
            Test_samp_merge = Test_samp_merge_drop.groupby('所属基站').apply(jifang_cnt)
            Test_samp_merge.drop_duplicates(subset=['所属基站'], keep='first', inplace=True)
            Inter = pd.merge(Test_samp_merge, Result_drop, how='left')
            Inter.dropna(axis=0, inplace=True)
            Inter1 = Inter[[
                '射频单元输入电源能力不足告警',
                'BBU直流输出异常告警', 'BBU IR光模块/电接口不在位告警',
                'BBU IR光模块故障告警', 'BBU IR光模块收发异常告警', '巡检重点关注项目'
            ]]  # '电源模块异常告警','市电输入异常告警','基站直流供电异常告警','单板输入电压异常告警','CXU输入电源能力不足告警
            add_cont = []
            columns_name = Test_samp1.columns.values.tolist()
            for idx in Inter1.index.values.tolist():
                st = ''
                typical_alarms = ['射频单元输入电源能力不足告警',
                                  'BBU直流输出异常告警', 'BBU IR光模块/电接口不在位告警',
                                  'BBU IR光模块故障告警', 'BBU IR光模块收发异常告警'
                                  ]  # '电源模块异常告警','市电输入异常告警','基站直流供电异常告警','单板输入电压异常告警','CXU输入电源能力不足告警
                for typical_alarm in typical_alarms:
                    if Inter1.loc[idx, typical_alarm] > 0 and typical_alarm not in Inter1.loc[idx, '巡检重点关注项目']:
                        st += "'({}){}',".format(columns_name.index(typical_alarm), typical_alarm)
                add_cont.append(st)
            Inter1['add_cont'] = add_cont
            print('完成-增加典型隐形告警名 in %s seconds' % (time.clock() - start))
            Inter['巡检重点关注项目'] = Inter['巡检重点关注项目'] + ',' + Inter1['add_cont']
            Inter['巡检重点关注项目'].map(lambda x: str(x).replace(';', ','))
            Fin_result = Inter[['巡检优先级', '基站健康度异常程度', '巡检重点关注项目', '所属基站', '备注', '推断模型', '告警时间戳']]
            Fin_result = Fin_result.sort_values(by='巡检优先级', ascending=True)
        else:
            Fin_result = Result_drop[['巡检优先级', '基站健康度异常程度', '巡检重点关注项目',  '重点关注影响业务告警发生频次','所属基站', 'enbid','备注', '推断模型','告警时间戳']]
        Fin_result = Fin_result.dropna()
        if os.path.exists(r"./Data/{}/Inspect_List/{}/".format(distr, ftype))==False:
            mkdir(r"./Data/{}/Inspect_List/{}/".format(distr, ftype))
        Fin_result = Fin_result.drop_duplicates(subset=['enbid'], keep='first')# 20210917有些基站一个ID对应多个基站中文名，需要去重
        Fin_result.to_csv(
            "./Data/{}/Inspect_List/{}/Origin_{}_4G.csv".format(distr, ftype, date),
            index=None,encoding='gbk')
        print('保存-巡检清单 in <./Data/{}/Inspect_List/{}/Origin_{}_4G.csv'.format(distr, ftype, date))
