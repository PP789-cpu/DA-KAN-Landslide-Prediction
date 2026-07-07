# DA-KAN Cross-regional Landslide Displacement Prediction

This repository contains the official implementation of the **Dynamically Activated Kolmogorov-Arnold Network (DA-KAN)** model, designed for cross-regional landslide displacement prediction. The framework leverages transfer learning to address the challenge of data scarcity in newly monitored landslide scenarios.

##  Project Overview

Predicting landslide displacement is critical for early warning systems. However, newly monitored landslide slopes often lack sufficient historical data for training robust deep learning models. This project introduces a transfer learning framework that pre-trains a DA-KAN model on a data-rich source domain (e.g., Baijiabao landslide) and fine-tunes it on a data-scarce target domain (e.g., Bazimen landslide).

The prediction process decomposes the cumulative displacement into:
1.  **Trend Component:** Represents the long-term geological evolution.
2.  **Periodic Component:** Represents short-term fluctuations driven by external factors (rainfall, reservoir water level).

## Repository Structure

The repository contains the following core files and datasets:

### Core Scripts & Directories
* **`tfkan/`**: A crucial directory containing the implementation of the Kolmogorov-Arnold Network (KAN) layers customized for TensorFlow/Keras. This directory must be present in the project root for the models to build successfully.
* **`1_source_trend_prediction.py`**: Script for pre-training the DA-KAN model on the trend component of the source domain.
* **`2_source_periodic_prediction.py`**: Script for pre-training the DA-KAN model on the periodic component of the source domain.
* **`3_target_transfer_trend.py`**: Script for fine-tuning the pre-trained trend model on the target domain using varying data ratios (Ensemble Transfer).
* **`4_target_transfer_periodic.py`**: Script for fine-tuning the pre-trained periodic model on the target domain using varying data ratios (Ensemble Transfer).
* **`DWT_Decomposition.py`**: Script for performing Discrete Wavelet Transform (DWT) to decompose raw cumulative displacement into trend and periodic components.

### Quick Evaluation (For Reviewers)
* **`quick_test.py`**: A fast-track validation script designed for reviewers. It executes the entire pipeline (Source Pre-training -> Target Fine-tuning -> Total Displacement Synthesis) using minimal epochs (e.g., 5 epochs) and prints evaluation metrics (R², RMSE) along with a visualization plot.

### Datasets
* **`source_sample.xlsx`**: A sample dataset containing the first 300 records from the source domain.
* **`target_sample.xlsx`**: The complete dataset for the target domain.

## Environment & Dependencies

Ensure you have Python 3.8+ installed. The following libraries are required:

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl xlsxwriter
pip install tensorflow 
