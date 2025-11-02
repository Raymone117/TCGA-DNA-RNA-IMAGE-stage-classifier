# TCGA-DNA-RNA-IMAGE-stage-classifier
Capstone-5703

DNA: This module focuses on the DNA modality for TCGA stage classification. The CNV data is preprocessed, feature-selected, and trained using an optimized XGBoost classifier, achieving test accuracy = 0.607 and macro F1 = 0.426.
The Train/ directory contains the training scripts, confusion-matrix visualizations, and the exported model (final_model.pkl).
The Client/ directory provides an updated federated learning client that connects to the central Flower server, evaluates the model locally, and uploads prediction results with modality-specific weighting.
Together, these components form the DNA branch of the multi-modal TCGA stage classification framework.

RNA:
This module focuses on the RNA modality for TCGA stage classification. RNA expression data were processed, feature-selected (SelectKBest), and class-balanced using SMOTE. The final model was trained using AutoGluon, achieving test accuracy = 0.68 and macro F1 = 0.67.
The Train/ folder contains the full training pipeline, preprocessing artifacts (scaler, selector), and the saved model.
The Client/ folder includes the federated client for local inference and JSON upload.
The Dataset/ folder provides raw RNA files and preprocessing scripts to generate .h5ad input.
This module enables RNA-based classification and contributes to the multi-modal stage fusion system.


WSI: This part focuses on the Image modality in the TCGA stage classification project.
Trained using Random Forest with tuned hyperparameters.
Achieved validation accuracy = 0.634, macro F1 = 0.474.
Added validation confusion matrices for performance visualization.
Applied class-weighting to automatically address class imbalance.
All random seeds fixed (RS = 42) to ensure consistent results.
randomwlak client updated to support the new model structure.
Main script: WSI/Train/wsi_randomwalk.ipynb.
output: stage_classifier (.pkl) , feat_cols (.pkl) and scaler (.pkl).
