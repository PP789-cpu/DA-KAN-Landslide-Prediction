import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from os.path import dirname
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanAbsoluteError as KAN_MAE
from tensorflow.keras.losses import MeanSquaredError as KAN_MSE

try:
    from tfkan.layers.dense import DenseKAN
except ImportError:
    DenseKAN = layers.Dense

CUSTOM_CLASSES = {'DenseKAN': DenseKAN}
try:
    tf.keras.utils.get_custom_objects().update(CUSTOM_CLASSES)
except Exception:
    pass

CUSTOM_OBJECTS_FOR_LOADING = {
    'DenseKAN': DenseKAN,
    'mse': KAN_MSE,
    'mae': KAN_MAE,
    'MeanAbsoluteError': KAN_MAE,
    'MeanSquaredError': KAN_MSE
}

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def create_time_series_dataset(data, lookback, horizon=1):
    X, Y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i + lookback)])
        Y.append(data[(i + lookback):(i + lookback + horizon)])
    return np.array(X)[..., np.newaxis], np.array(Y)

def build_dakan_model(lookback_window, input_dim=1, output_dim=1):
    CONV_FILTERS = 50
    CONV_KERNEL = 3
    DILATIONS = [1, 2]
    KAN_GRID_SIZE = 7
    KAN_SPLINE_ORDER = 3

    inputs = Input(shape=(lookback_window, input_dim), name='Input_Sequence')
    x = inputs

    for d in DILATIONS:
        x = layers.Conv1D(
            filters=CONV_FILTERS,
            kernel_size=CONV_KERNEL,
            padding='causal',
            dilation_rate=d,
            activation='relu'
        )(x)

    conv_output = layers.GlobalAveragePooling1D()(x)

    kan_output = DenseKAN(
        units=CONV_FILTERS * 2,
        grid_size=KAN_GRID_SIZE,
        spline_order=KAN_SPLINE_ORDER,
        use_bias=True
    )(conv_output)

    dropout = layers.Dropout(0.1)(kan_output)
    outputs = layers.Dense(units=output_dim, activation='linear')(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model

def run_trend_prediction_workflow(input_file, sheet_name, data_column, date_column, output_file, model_save_path, lookback=3, train_ratio=0.8):
    try:
        df_data = pd.read_excel(input_file, sheet_name=sheet_name)

        raw_data = df_data[data_column].values.astype(float).reshape(-1, 1)

        scaler_Y = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler_Y.fit_transform(raw_data)

        X_norm, Y_norm = create_time_series_dataset(normalized_data.flatten(), lookback=lookback, horizon=1)

        df_timestamp = df_data[date_column].iloc[lookback:].reset_index(drop=True)

        total_samples = len(X_norm)
        train_size = int(train_ratio * total_samples)

        X_train, X_test = X_norm[:train_size], X_norm[train_size:]
        Y_train, Y_test = Y_norm[:train_size], Y_norm[train_size:]

        timestamp_test = df_timestamp.iloc[train_size:]

        input_dim = X_train.shape[-1]
        model = build_dakan_model(lookback_window=lookback, input_dim=input_dim, output_dim=1)

        model.fit(X_train, Y_train, epochs=200, batch_size=32, verbose=2)

        model_dir = dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_save_path)

        Y_pred_norm = model.predict(X_test)
        Y_true = scaler_Y.inverse_transform(Y_test)
        Y_pred = scaler_Y.inverse_transform(Y_pred_norm)

        metrics = calculate_metrics(Y_true, Y_pred)

        plt.figure(figsize=(10, 5))
        plt.plot(Y_true.flatten(), label='Actual Trend', color='blue', linestyle='-')
        plt.plot(Y_pred.flatten(), label='Predicted Trend', color='red', linestyle='--')
        plt.title(f'Source Domain Trend Prediction (R2: {metrics["R2"]:.4f})')
        plt.xlabel('Time Steps')
        plt.ylabel('Displacement')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        output_dir = dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df_results = pd.DataFrame({
            'Time': timestamp_test.values,
            'Actual_Trend_Value': Y_true.flatten(),
            'Predicted_Trend_Value': Y_pred.flatten()
        })
        df_metrics = pd.DataFrame([metrics])

        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, sheet_name='Trend_Prediction_Results', index=False)
            df_metrics.to_excel(writer, sheet_name='Performance_Metrics', index=False)

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    INPUT_FILE = 'source_sample.xlsx'
    SHEET_NAME = 'Sheet1'
    DATA_COLUMN = 'Trend'
    DATE_COLUMN = 'Time'
    OUTPUT_FILE = 'tcn_kan_prediction_results_trend1.xlsx'
    MODEL_SAVE_PATH = 'tcn_kan_zd1_trend_model.h5'
    LOOKBACK_WINDOW = 3

    run_trend_prediction_workflow(
        input_file=INPUT_FILE,
        sheet_name=SHEET_NAME,
        data_column=DATA_COLUMN,
        date_column=DATE_COLUMN,
        output_file=OUTPUT_FILE,
        model_save_path=MODEL_SAVE_PATH,
        lookback=LOOKBACK_WINDOW
    )