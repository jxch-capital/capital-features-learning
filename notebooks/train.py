import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

import api.capital_features_api as cf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
import json


def to_dataset(train_data, validation_data, Y_train, Y_val, batch=64):
    scaler = StandardScaler()
    X_train = np.array(train_data['featuresT'])
    X_val = np.array(validation_data['featuresT'])
    # 获取形状信息
    num_samples, num_timesteps, num_features = X_train.shape
    # 将三维特征数组重塑为二维
    X_train_reshaped = X_train.reshape(-1, num_features)
    # 使用训练集的数据来拟合scaler
    scaler.fit(X_train_reshaped)

    # 标准化训练数据
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(num_samples, num_timesteps, num_features)
    # 同样的，将验证集（如果有的话）重塑并转换
    num_samples_val, num_timesteps_val, num_features_val = X_val.shape
    X_val_reshaped = X_val.reshape(-1, num_features_val)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(num_samples_val, num_timesteps_val, num_features_val)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, Y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, Y_val))
    BATCH_SIZE = batch  # 你可以根据需要调整这个值
    train_dataset = train_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    return train_dataset, validation_dataset, scaler


def to_prediction_scaled(prediction_data, scaler):
    X_prediction = np.array(prediction_data['knodeTrains']['featuresT'])
    num_samples_prediction, num_timesteps_prediction, num_features_prediction = X_prediction.shape
    X_prediction_reshaped = X_prediction.reshape(-1, num_features_prediction)
    X_prediction_scaled = scaler.transform(X_prediction_reshaped).reshape(num_samples_prediction,
                                                                          num_timesteps_prediction,
                                                                          num_features_prediction)
    return X_prediction_scaled


def to_weights(Y_train):
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(Y_train),
        y=Y_train
    )
    return dict(zip(np.unique(Y_train), weights))


def max_acc(y_true, y_pred):
    # 设定一个阈值来确定分类（例如，0.5）
    threshold = 0.5
    y_pred = tf.cast(y_pred >= threshold, tf.float32)

    # 这里，我们将False Negatives的权重设为0，也就是说它们不计入总的损失
    fn_weight = 0.2
    fp_weight = 1.0  # False Positives的权重
    tn_weight = 1.0  # True Negatives的权重
    tp_weight = 1.0  # True Positives的权重

    # 计算不同类型错误的数量
    tp = K.sum(tp_weight * y_true * y_pred)
    fp = K.sum(fp_weight * (1 - y_true) * y_pred)
    fn = K.sum(fn_weight * y_true * (1 - y_pred))
    tn = K.sum(tn_weight * (1 - y_true) * (1 - y_pred))

    # 计算自定义评估指标
    # 在这个例子中，我们只计算被正确分类的样本的比例作为评估指标
    custom_score = (tp + tn) / (tp + fp + fn + tn)
    return custom_score


def get_model(x_shape=5, y_shape=40, model_name="train"):
    model = Sequential([
        tf.keras.layers.InputLayer(input_shape=(x_shape, y_shape)),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )

    # 创建EarlyStopping和ReduceLROnPlateau回调
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=20,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    # 创建ModelCheckpoint回调
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoint/' + model_name + '/epoch_{epoch:02d}.ckpt',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
    )

    # 返回模型和回调列表
    return model, [early_stopping, reduce_lr, model_checkpoint]


def save_scaler(path, scaler):
    mean_up = scaler.mean_
    var_up = scaler.var_
    with open(path, 'w') as f_out:
        json.dump({'mean': mean_up.tolist(), 'var': var_up.tolist()}, f_out)
