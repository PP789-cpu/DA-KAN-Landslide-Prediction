import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from os.path import dirname

def plot_results(original_data, modes, mode_names, title_prefix="Mode"):
    num_modes = modes.shape[0]

    fig, axes = plt.subplots(num_modes + 1, 1, figsize=(10, 2 * (num_modes + 1)))
    plt.subplots_adjust(hspace=0.4)

    axes[0].plot(original_data, 'r')
    axes[0].set_title('Original Displacement Data', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    for i in range(num_modes):
        axes[i + 1].plot(modes[i, :], 'k')
        axes[i + 1].set_title(f'{title_prefix} {mode_names[i]}', fontsize=10)
        axes[i + 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    print("\n[Plotting] Chart displayed. Close the window to continue.")
    plt.show()

def dwt_decomposition_and_save(input_file_path, sheet_name, data_column_name, output_file_path, plot_results_flag=False):
    try:
        print(f"Reading data from: {input_file_path} (Sheet: {sheet_name})...")

        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"File not found: {input_file_path}")

        df_original = pd.read_excel(input_file_path, sheet_name=sheet_name)

        if data_column_name not in df_original.columns:
            raise ValueError(f"Column '{data_column_name}' not found in the dataset.")

        y = np.array(df_original[data_column_name].values, dtype=float)

        wavelet_name = 'db4'
        decomposition_level = 6

        print(f"Performing DWT decomposition (Wavelet: {wavelet_name}, Level: {decomposition_level})...")

        coeffs = pywt.wavedec(y, wavelet_name, level=decomposition_level)

        reconstructed_modes = []
        mode_names = []

        cA_N = coeffs[0]
        A_N = pywt.waverec([cA_N] + [None] * decomposition_level, wavelet_name)[:len(y)]
        reconstructed_modes.append(A_N)
        mode_names.append(f'A_{decomposition_level}')

        for i in range(1, decomposition_level + 1):
            cD_i = coeffs[i]
            cD_list = [np.zeros_like(c) for c in coeffs]
            cD_list[0] = np.zeros_like(cA_N)
            cD_list[i] = cD_i

            detail_level = decomposition_level - i + 1
            mode_names.append(f'D_{detail_level}')

            D_i = pywt.waverec(cD_list, wavelet_name)[:len(y)]
            reconstructed_modes.append(D_i)

        modes_array = np.vstack(reconstructed_modes)

        reconstructed_signal = np.sum(modes_array, axis=0)
        rmse = np.sqrt(np.mean((reconstructed_signal - y) ** 2))
        print(f"Reconstruction RMSE: {rmse:.12f}")

        results_dict = {}
        for name, mode in zip(mode_names, reconstructed_modes):
            results_dict[name] = mode

        df_results = pd.DataFrame(results_dict)
        df_results.insert(0, 'Original_Data', y)

        output_dir = dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Decomposition complete. Saving results to: {output_file_path}...")
        df_results.to_excel(output_file_path, index=False)

        print("\nExecution successful!")
        print(f"Total components generated: {decomposition_level + 1}")

        if plot_results_flag:
            print("Plotting DWT decomposition results...")
            plot_results(y, modes_array, mode_names, title_prefix="Wavelet")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    INPUT_FILE = './data/sample_data.xlsx'
    SHEET_NAME = 'Sheet1'
    DATA_COLUMN = 'Total'
    OUTPUT_FILE = './output/dwt_results.xlsx'

    dwt_decomposition_and_save(
        input_file_path=INPUT_FILE,
        sheet_name=SHEET_NAME,
        data_column_name=DATA_COLUMN,
        output_file_path=OUTPUT_FILE,
        plot_results_flag=True
    )