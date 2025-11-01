# TCGA-DNA-RNA-IMAGE-stage-classifier
Capstone-5703

DNA: This part focuses on the DNA modality in the TCGA stage classification project.
Trained using XGBoost with tuned hyperparameters
Achieved test accuracy = 0.607, macro F1 = 0.426
Added validation and test confusion matrices for performance visualization
Applied Stage III weight = 1.3 to address class imbalance
All random seeds fixed (RS = 192) to ensure consistent results
Client 2.0 updated to support the new model structure
Main script: DNA/Train/DNA_XGBOOST5.0.py
Output: trained model (.pkl) and confusion matrices (.png)


WSI: This part focuses on the Image modality in the TCGA stage classification project.
Trained using Random Forest with tuned hyperparameters
Achieved validation accuracy = 0.634, macro F1 = 0.474
Added validation confusion matrices for performance visualization
Applied class-weighting to automatically address class imbalance
All random seeds fixed (RS = 42) to ensure consistent results
randomwlak client updated to support the new model structure
Main script: WSI/Train/wsi_randomwalk.ipynb
output: stage_classifier (.pkl) , feat_cols (.pkl) and scaler (.pkl)
