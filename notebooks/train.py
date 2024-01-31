import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

import api.capital_features_api as cf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
import json
import tf_util as tu


def to_test_dataset(test_data, y_test, scaler, batch_size=64):
    x_test = np.array(test_data['featuresT'])
    num_samples, num_timesteps, num_features = x_test.shape
    x_test_reshaped = x_test.reshape(-1, num_features)
    x_test_scaled = scaler.transform(x_test_reshaped).reshape(num_samples, num_timesteps, num_features)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_scaled, y_test))
    test_dataset = test_dataset.batch(batch_size)
    return test_dataset


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
        patience=6,
        min_lr=1e-6,
        verbose=1
    )

    # 创建ModelCheckpoint回调
    model_checkpoint = ModelCheckpoint(
        filepath='./epoch/' + model_name + '/epoch_{epoch:02d}.ckpt',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
    )

    csv_logger = CSVLogger('./log/' + model_name + '.csv', append=True)

    # 返回模型和回调列表
    return model, [early_stopping, reduce_lr, model_checkpoint, csv_logger]


import sys


def fit2log_txt(model, train_dataset, validation_dataset, callbacks, weights, log_path):
    with open(log_path, 'a') as log_file:
        original_stdout = sys.stdout
        try:
            sys.stdout = log_file
            his = model.fit(train_dataset, epochs=1000, validation_data=validation_dataset,
                            verbose=2, callbacks=callbacks, class_weight=weights,
                            initial_epoch=tu.find_last_epoch_txt_log(log_path) + 1)
        finally:
            sys.stdout = original_stdout

    return his


def save_scaler(scaler, path):
    mean_up = scaler.mean_
    var_up = scaler.var_
    with open(path, 'w') as f_out:
        json.dump({'mean': mean_up.tolist(), 'var': var_up.tolist()}, f_out)


class CustomSelectiveAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_selective_accuracy', threshold=0.5, **kwargs):
        super(CustomSelectiveAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct_count = self.add_weight(name='correct_count', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 根据阈值确定是否给出确定预测
        y_pred_classes = tf.where(y_pred >= self.threshold, 1.0, 0.0)
        y_true_classes = tf.cast(y_true, 'float32')

        # 根据确定性阈值筛选出确定的预测
        is_determined = tf.logical_or(y_pred >= self.threshold, y_pred <= (1 - self.threshold))

        # 计算确定预测中准确的数量
        correct_predictions = tf.cast(tf.equal(y_pred_classes, y_true_classes), 'float32') * tf.cast(is_determined,
                                                                                                     'float32')

        self.correct_count.assign_add(tf.reduce_sum(correct_predictions))
        self.total_count.assign_add(tf.reduce_sum(tf.cast(is_determined, 'float32')))

    def result(self):
        return self.correct_count / self.total_count

    def reset_state(self):
        self.correct_count.assign(0.)
        self.total_count.assign(0.)

    def get_correct_predictions(self):
        return self.correct_count


def evaluate(thresholds, model_ud, test_dataset):
    # 准备收集准确率和损失值
    accuracies = []
    losses = []
    correct_predictions = []

    for thresh in thresholds:
        # 定义一个新的指标实例
        custom_acc = CustomSelectiveAccuracy(threshold=thresh)

        # 使用自定义的准确率指标来编译模型
        model_ud.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=[custom_acc])

        # 评估模型
        result = model_ud.evaluate(test_dataset, verbose=1)

        # 收集损失值和准确率
        losses.append(result[0])
        accuracies.append(result[1])
        correct_predictions.append(custom_acc.get_correct_predictions())

    return {
        'accuracies': accuracies,
        'losses': losses,
        'correct_predictions': correct_predictions,
    }
