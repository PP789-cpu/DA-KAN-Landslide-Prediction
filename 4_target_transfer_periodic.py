import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanAbsoluteError as KAN_MAE
from tensorflow.keras.losses import MeanSquaredError as KAN_MSE

try:
    from tfkan.layers.dense import DenseKAN
except ImportError:
    DenseKAN = layers.Dense

CUSTOM_OBJECTS = {
    'DenseKAN': DenseKAN,
    'mse': KAN_MSE,
    'mae': KAN_MAE,
    'MeanAbsoluteError': KAN_MAE,
    'MeanSquaredError': KAN_MSE
}
tf.keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONFIG = {
    'LOOKBACK': 1,
    'BATCH_SIZE_TGT': 10,
    'EXTERNAL_FEAT_COL': ['Reservoir', 'Rainfall'],
    'TARGET_COL': 'Periodic',
    'TARGET_PATH': 'target_sample.xlsx',
    'PRETRAINED_MODEL_PATH': 'tcn_kan_zd1_periodic_model.h5',
    'RESULT_XLS': 'ZG111_Cycle_Transfer_Final.xlsx'
}

def create_dataset(data, target, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), :])
        Y.append(target[i + lookback])
    return np.array(X), np.array(Y)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def train_single_transfer_model(X_train, Y_train, X_test, Y_test, seed, run_id):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = load_model(CONFIG['PRETRAINED_MODEL_PATH'], custom_objects=CUSTOM_OBJECTS, compile=False)

    for layer in model.layers:
        layer.trainable = True

    temp_path = f"temp_periodic_{run_id}.weights.h5"
    if os.path.exists(temp_path): os.remove(temp_path)

    cbs = [
        ModelCheckpoint(temp_path, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min'),
        ReduceLROnPlateau(factor=0.7, patience=8, min_lr=1e-6),
        EarlyStopping(patience=20, restore_best_weights=True)
    ]

    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), loss='mse', metrics=['mae'])

    mini_batch = min(len(X_train), 8)

    model.fit(X_train, Y_train, epochs=300, batch_size=mini_batch,
              validation_data=(X_test, Y_test), callbacks=cbs, verbose=0)

    if os.path.exists(temp_path):
        model.load_weights(temp_path)

    pred = model.predict(X_test, verbose=0)

    if os.path.exists(temp_path): os.remove(temp_path)
    return pred

def run_transfer_target():
    try:
        if not os.path.exists(CONFIG['TARGET_PATH']):
            return
        if not os.path.exists(CONFIG['PRETRAINED_MODEL_PATH']):
            return

        df = pd.read_excel(CONFIG['TARGET_PATH'])

        lag_col_name = 'lag_periodic'
        df[lag_col_name] = df[CONFIG['TARGET_COL']].shift(1)
        df = df.dropna(subset=[lag_col_name]).reset_index(drop=True)

        final_features = [lag_col_name] + CONFIG['EXTERNAL_FEAT_COL']

        feature_data = df[final_features].values
        target_data = df[CONFIG['TARGET_COL']].values.reshape(-1, 1)
        time_data = df['Time'].values if 'Time' in df.columns else np.arange(len(df))

        scaler_X = MinMaxScaler((0, 1))
        X_scaled = scaler_X.fit_transform(feature_data)
        scaler_Y = MinMaxScaler((0, 1))
        Y_scaled = scaler_Y.fit_transform(target_data)

        X, Y = create_dataset(X_scaled, Y_scaled, CONFIG['LOOKBACK'])
        Y = Y.reshape(len(Y), -1)

        total_samples = len(X)
        test_start_idx = int(total_samples * 0.7)

        ensure_dir(CONFIG['RESULT_XLS'])

        ratios = [0.1, 0.3, 0.5, 0.7]
        summary_metrics = []

        with pd.ExcelWriter(CONFIG['RESULT_XLS'], engine='openpyxl') as writer:
            for r in ratios:
                train_end_idx = int(total_samples * r)
                if train_end_idx >= test_start_idx:
                    train_end_idx = test_start_idx - 1

                X_train, Y_train = X[:train_end_idx], Y[:train_end_idx]
                X_test, Y_test = X[test_start_idx:], Y[test_start_idx:]
                time_test = time_data[CONFIG['LOOKBACK']:][test_start_idx:]

                preds = []
                n_ensemble = 8
                for i in range(n_ensemble):
                    p = train_single_transfer_model(X_train, Y_train, X_test, Y_test,
                                                    seed=100 + i, run_id=f"Cycle_R{r}_{i}")
                    preds.append(p)

                avg_pred_scaled = np.mean(preds, axis=0)

                Y_test_real = scaler_Y.inverse_transform(Y_test)
                Y_pred_real = scaler_Y.inverse_transform(avg_pred_scaled)

                r2 = r2_score(Y_test_real, Y_pred_real)
                rmse = np.sqrt(mean_squared_error(Y_test_real, Y_pred_real))
                mae = mean_absolute_error(Y_test_real, Y_pred_real)

                summary_metrics.append({'Train_Ratio': r, 'R2': r2, 'RMSE': rmse, 'MAE': mae})

                df_res = pd.DataFrame({
                    'Time': time_test,
                    'Actual': Y_test_real.flatten(),
                    'Predicted': Y_pred_real.flatten()
                })
                df_res.to_excel(writer, sheet_name=f'Ratio_{r}', index=False)

            if summary_metrics:
                df_sum = pd.DataFrame(summary_metrics)
                df_sum.to_excel(writer, sheet_name='Summary_Metrics', index=False)

            plt.figure(figsize=(10, 5))
            plt.plot(Y_test_real.flatten(), label='Actual Periodic', color='blue', linestyle='-')
            plt.plot(Y_pred_real.flatten(), label='Predicted Periodic', color='red', linestyle='--')
            plt.title(f'Target Domain Periodic Prediction (R2: {r2:.4f})')
            plt.xlabel('Time Steps')
            plt.ylabel('Displacement')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_transfer_target()