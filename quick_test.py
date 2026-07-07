import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanAbsoluteError as KAN_MAE
from tensorflow.keras.losses import MeanSquaredError as KAN_MSE

try:
    from tfkan.layers.dense import DenseKAN
except ImportError:
    DenseKAN = layers.Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FAST_EPOCHS = 5
BATCH_SIZE = 8

CONFIG = {
    'ZD1_PATH': 'source_sample.xlsx',
    'ZG111_PATH': 'target_sample.xlsx',
    'TREND_LB': 3,
    'Periodic_LB': 1,
    'TREND_MODEL_PATH': 'temp_fast_Trend_src.h5',
    'Periodic_MODEL_PATH': 'temp_fast_Periodic_src.h5'
}

CUSTOM_OBJECTS = {
    'DenseKAN': DenseKAN,
    'mse': KAN_MSE, 'mae': KAN_MAE,
    'MeanAbsoluteError': KAN_MAE, 'MeanSquaredError': KAN_MSE
}
tf.keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)


def create_dataset(data, target, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), :])
        Y.append(target[i + lookback])
    return np.array(X), np.array(Y)


def build_dakan_model(lookback_window, input_dim):
    inputs = Input(shape=(lookback_window, input_dim))
    x = inputs
    for d in [1, 2]:
        x = layers.Conv1D(filters=50, kernel_size=3, padding='causal', dilation_rate=d, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = DenseKAN(units=100, grid_size=7, spline_order=3, use_bias=True)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(units=1, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model


def fast_pretrain(data_path, feat_cols, target_col, lb, model_save_path, is_Periodic=False):
    df = pd.read_excel(data_path)

    if is_Periodic:
        lag_col = 'lag_periodic'
        df[lag_col] = df[target_col].shift(1)
        df = df.dropna(subset=[lag_col]).reset_index(drop=True)
        feat_cols = [lag_col] + feat_cols

    X_raw = df[feat_cols].values.astype(float)
    Y_raw = df[target_col].values.astype(float).reshape(-1, 1)

    scaler_X = MinMaxScaler().fit(X_raw)
    scaler_Y = MinMaxScaler().fit(Y_raw)

    X, Y = create_dataset(scaler_X.transform(X_raw), scaler_Y.transform(Y_raw), lb)

    model = build_dakan_model(lb, len(feat_cols))
    model.fit(X, Y, epochs=FAST_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    model.save(model_save_path)


def fast_transfer(data_path, feat_cols, target_col, lb, pretrained_model_path, is_Periodic=False):
    df = pd.read_excel(data_path)

    if is_Periodic:
        lag_col = 'lag_periodic'
        df[lag_col] = df[target_col].shift(1)
        df = df.dropna(subset=[lag_col]).reset_index(drop=True)
        feat_cols = [lag_col] + feat_cols

    X_raw = df[feat_cols].values.astype(float)
    Y_raw = df[target_col].values.astype(float).reshape(-1, 1)
    time_series = df['时间'].values if '时间' in df.columns else np.arange(len(df))

    scaler_X = MinMaxScaler().fit(X_raw)
    scaler_Y = MinMaxScaler().fit(Y_raw)

    X, Y = create_dataset(scaler_X.transform(X_raw), scaler_Y.transform(Y_raw), lb)

    split_idx = int(len(X) * 0.7)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    test_dates = time_series[lb:][split_idx:]

    model = load_model(pretrained_model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=optimizers.Adam(1e-4), loss='mse')
    model.fit(X_train, Y_train, epochs=FAST_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    pred_norm = model.predict(X_test, verbose=0)

    y_true = scaler_Y.inverse_transform(Y_test).flatten()
    y_pred = scaler_Y.inverse_transform(pred_norm).flatten()

    return y_true, y_pred, test_dates


if __name__ == "__main__":
    fast_pretrain(CONFIG['ZD1_PATH'], ['Trend'], 'Trend', CONFIG['TREND_LB'], CONFIG['TREND_MODEL_PATH'], is_Periodic=False)
    fast_pretrain(CONFIG['ZD1_PATH'], ['Reservoir', 'Rainfall'], 'Periodic', CONFIG['Periodic_LB'], CONFIG['Periodic_MODEL_PATH'],
                  is_Periodic=True)

    true_trend, pred_trend, time_trend = fast_transfer(
        CONFIG['ZG111_PATH'], ['Trend'], 'Trend', CONFIG['TREND_LB'], CONFIG['TREND_MODEL_PATH'], is_Periodic=False)

    true_Periodic, pred_Periodic, time_Periodic = fast_transfer(
        CONFIG['ZG111_PATH'], ['Reservoir', 'Rainfall'], 'Periodic', CONFIG['Periodic_LB'], CONFIG['Periodic_MODEL_PATH'], is_Periodic=True)

    min_len = min(len(pred_trend), len(pred_Periodic))

    final_true = true_trend[-min_len:] + true_Periodic[-min_len:]
    final_pred = pred_trend[-min_len:] + pred_Periodic[-min_len:]
    final_time = time_trend[-min_len:]

    r2 = r2_score(final_true, final_pred)

    plt.figure(figsize=(10, 5))
    plt.plot(final_true, label='Total Actual', color='blue', linestyle='-', linewidth=2)
    plt.plot(final_pred, label='Total Predicted', color='red', linestyle='--', linewidth=2)
    plt.title(f'Target Domain Total Displacement Transfer Prediction (R2: {r2:.2f})')
    plt.xlabel('Test Time Steps')
    plt.ylabel('Cumulative Displacement')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    if os.path.exists(CONFIG['TREND_MODEL_PATH']): os.remove(CONFIG['TREND_MODEL_PATH'])
    if os.path.exists(CONFIG['Periodic_MODEL_PATH']): os.remove(CONFIG['Periodic_MODEL_PATH'])