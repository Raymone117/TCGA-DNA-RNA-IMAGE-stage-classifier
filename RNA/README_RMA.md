# RNA Modality Pipeline for Cancer Stage Classification (TCGA-BRCA)

This repository contains the complete RNA modality pipeline for the TCGA-BRCA multimodal cancer stage classification project. The goal is to process raw RNA-seq expression data, train a robust AutoGluon-based classifier, and generate standardized prediction outputs for server-side fusion in a federated learning setting.

The pipeline is modularized into three main stages: dataset preparation, model training, and client-side inference.

## Directory Structure Overview
```
RNA/
├── Client/                    # Client-side inference and JSON output module
│   └── RNAClient.ipynb         # Main script for test prediction and JSON formatting
├── Dataset/                   # Raw data and preprocessing scripts
│   ├── OriginalData_from_GDC/  # RNA expression files downloaded from GDC
│   ├── Data_results/           # Intermediate outputs (.h5ad format, etc.)
│   └── DataPrepare.ipynb       # Script to convert raw RNA data to .h5ad format
├ ── Train/                     # Training and validation workflow
│   ├── autogluon_rna_model/    # AutoGluon model checkpoint directory
│   ├── RNA_model.ipynb         # Main training pipeline: preprocessing + training
│   ├── RNA_scaler.pkl          # Saved StandardScaler
│   ├── RNA_selector_kbest.pkl  # Saved SelectKBest feature selector
│   ├── RNA_test.h5ad           # Processed RNA test set
│   └── test_metadata.csv       # Metadata file with stage labels
```

## Module Descriptions

### 1. RNA Dataset Preparation (`Dataset/DataPrepare.ipynb`)

- Downloads RNA expression data (TSV format) from GDC
- Aligns sample IDs with clinical labels via metadata
- Converts merged expression matrix into `.h5ad` format for downstream modeling (compatible with Scanpy / AnnData)

### 2. Data Preprocessing and Model Training (`Train/RNA_model.ipynb`)

- **Standardization**: Uses `StandardScaler` to normalize gene expression values
- **Feature Selection**: Applies `SelectKBest` to retain top 500 stage-related genes
- **SMOTE Oversampling**: Addresses class imbalance (especially Stage IV) on the training set
- **Model Training (AutoGluon)**: Automatically ensembles multiple classifiers (GBM, CatBoost, XGBoost, RandomForest)
- Saves model artifacts (scaler, selector, trained model) for later inference

### 3. Inference and Output Client (`Client/RNAClient.ipynb`)

- Loads test `.h5ad` data and applies the saved scaler and feature selector
- Predicts stage labels using AutoGluon ensemble model
- Exports a standardized JSON file for federated server consumption, including:
  - `patient_id`
  - `probs` (softmax output)
  - `modality`: set to `"rna"`
  - `weight`: configurable modality weight

## Evaluation Summary

| Dataset   | Accuracy | Macro F1 | Weighted F1 |
|-----------|----------|----------|-------------|
| Validation Set | 0.74     | 0.54     | 0.72        |
| Test Set       | 0.67     | 0.51     | 0.66        |

## Recommended Execution Order

1. Run `Dataset/DataPrepare.ipynb` to convert raw TSV into `.h5ad`
2. Run `Train/RNA_model.ipynb` to preprocess data, train the model, and save artifacts
3. Run `Client/RNAClient.ipynb` to generate softmax predictions and JSON outputs for server-side fusion

## Dependencies

This pipeline requires the following libraries:

- `pandas`, `numpy`
- `scanpy`, `anndata`
- `scikit-learn`
- `imbalanced-learn`
- `AutoGluon`
- `joblib`

Ensure these are installed in your Python environment before running the scripts.
