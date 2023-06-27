import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import keras
import matplotlib
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model,Sequential,load_model
from keras.layers import Dense, Input,Dropout,Lambda
import matplotlib.pyplot as plt
from keras import regularizers
from mpl_toolkits.mplot3d import Axes3D
from keras.optimizers import Adam,RMSprop
from keras import losses
from keras import initializers
from utils import mkdir

def data_loader(train_date, distr,ftype):
    """
    :param train_date: 例如："20200120-20200801"
    :param distr:
    :param ftype:
    :return:
    """
    X_Zeros = 1000  # 增加1000全0行
    Train_List = []
    try:
        train_data = pd.read_csv('./Data/{}/Alert_Samp/Samp_train_XJ_{}/故障_样本_{}_4G.csv'.format(distr,ftype, train_date),
                                     encoding='gbk',engine='python',index_col=0)  # index_col=0)
    except:
        train_data = pd.read_csv(r'./Data/{}/Alert_Samp/Samp_train_XJ_{}/故障_样本_{}_4G.csv'.format(distr,ftype,train_date),
                                     encoding='utf-8', engine='python',index_col=0)  # index_col=0)

    x_zeros = np.zeros((X_Zeros, len(train_data.iloc[0])))
    HW_zeros = pd.DataFrame(x_zeros, columns=train_data.columns)
    HW_Sample_add0 = pd.concat([train_data, HW_zeros], axis=0)
    x_train1 = HW_Sample_add0.values
    # 标准化
    scaled = np.copy(x_train1)
    ss = preprocessing.MinMaxScaler()
    import pickle
    x_train = ss.fit_transform(scaled)
    mkdir(r'./Data/{}/Inter_Result/4G/{}_故障_标准化/'.format(distr,ftype))
    with open(r'./Data/{}/Inter_Result/4G/{}_故障_标准化/标准化_{}.pk'.format(distr,ftype,train_date), 'wb') as f:
        pickle.dump(ss, f)
        f.close()
    return x_train
def ae_model(x_train,ftype,distr,date):
    Diff_max = 350  # 取前300误差最大的行
    Diff_row_max = 8  # 取前8个误差最大的维度
    Diff_threshold = 0.97
    # ———————————————————— 模型 ——————————————————————#
    original_dim = len(x_train[0])
    hidden_1 = round(original_dim / 2)
    hidden_2 = round(original_dim / 4)
    batch_size = 256
    encod_dim = 16  # 画2维图
    epoch = 100
    PRE_all = 1  # 0为预测整体  1为预测部分
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.01, patience=5, mode="min")

    window_size = len(x_train[0])
    print(window_size)
    # Build autoencoder.
    inputs = Input(shape=(window_size,))
    encoded = (Dense(hidden_1, activation='relu', kernel_initializer='glorot_normal'))(inputs)
    encoded = (Dense(hidden_2, activation='relu', kernel_initializer='glorot_normal'))(encoded)
    encoded = (Dense(encod_dim, activation='relu', kernel_initializer='glorot_normal'))(encoded)
    decoded = (Dense(hidden_2, activation='relu', kernel_initializer='glorot_normal'))(encoded)
    decoded = (Dense(hidden_1, activation='relu', kernel_initializer='glorot_normal'))(decoded)
    decoded = (Dense(window_size))(decoded)
    # decoded = (Dense(window_size, activation='selu'))(decoded)
    # Compile model and train.
    adadelta = keras.optimizers.Adadelta()
    encoder = Model(inputs, encoded)
    Ate = Model(inputs, decoded)
    Ate.summary()
    Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    def mean_squa(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
    Ate.compile(optimizer=Nadam, loss=mean_squa)
    history = Ate.fit(x_train, x_train,
                       epochs=epoch,
                       verbose=2,
                       batch_size=batch_size,
                       shuffle=True)
    mkdir(r"./Data/{}/Inter_Result/4G/{}_故障_模型/".format(distr,ftype))
    Ate.save(r"./Data/{}/Inter_Result/4G/{}_故障_模型/模型AE_{}.h5".format
        (distr,ftype,date))
def vae_model(x_train,ftype,distr,date):
    original_dim = len(x_train[0]) # 173
    batch_size = 256
    latent_dim = 16  # 隐变量取2维只是为了方便后面画图
    intermediate_dim_1 = round(original_dim / 2)
    intermediate_dim_2 = round(original_dim / 4)
    epochs = 100
    # ,kernel_initializer=initializers.lecun_normal(seed=None)
    inputs = Input(shape=(original_dim,))
    h1 = Dense(intermediate_dim_1, activation='relu', kernel_initializer='RandomUniform')(
        inputs)
    h2 = Dense(intermediate_dim_2, activation='relu', kernel_initializer='RandomUniform')(h1)

    # 算p(Z|X)的均值和方差
    z_mean = Dense(latent_dim)(h2)
    #    epsilon = K.random_no(h2)
    z_log_var = Dense(latent_dim)(h2)

    def sampling(args):
        z_mean, z_log_var = args
        # 返回具有正态值分布的张量。
        epsilon_std = 1.0
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]), mean=0.0,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # 重参数层，相当于给输入加入噪声
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_h1 = Dense(intermediate_dim_2, activation='relu')(z)
    decoder_h2 = Dense(intermediate_dim_1, activation='relu')(decoder_h1)

    outputs = Dense(original_dim, activation='selu')(decoder_h2)

    vae = Model(inputs, outputs)

    def vae_loss(inputs, outputs):
        # reconstruction_loss是重构loss，kl_loss是KL loss
        reconstruction_loss = losses.mse(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean+1e-10) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    adam = Adam(lr=0.0001) # 原来 0.001
    vae.compile(optimizer=adam, loss=vae_loss)
    vae.summary()
    history = vae.fit(x_train, x_train,
                      shuffle=True,
                      epochs=epochs,
                      verbose=2,
                      batch_size=batch_size,
                      )# validation_split=0.2)
    # --------SaveModel--------
    mkdir(r"./Data/{}/Inter_Result/4G/{}_故障_模型/".format(distr,ftype))
    vae.save(r"./Data/{}/Inter_Result/4G/{}_故障_模型/模型VAE_{}.h5".format(distr,ftype,date))

def train_XJ(para):
    train_date = para.train_date
    distr_list = para.distr_list
    ftype = para.ftype
    for distr in distr_list:
        data = data_loader(train_date, distr,ftype)
        ae_model(data,ftype,distr,train_date)
        vae_model(data,ftype,distr,train_date)
