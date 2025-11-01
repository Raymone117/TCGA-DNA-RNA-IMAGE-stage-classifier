"""
============================================================
Model Training & Evaluation (XGBoost)
This script trains an XGBoost classifier on the processed DNA CNV dataset.
It performs manual hyperparameter search over 15 parameter sets,
evaluates validation and test performance, and saves the final model bundle.
============================================================
"""

import os, joblib, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RS = 192
np.random.seed(RS)
os.environ["PYTHONHASHSEED"] = str(RS)
TRAIN_PATH = "DNA_train.csv"
TEST_PATH  = "DNA_test.csv"
SAVE_DIR   = "models_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Manual Parameter Grid (15 candidate configurations)
# ============================================================
param_list = [
    # --- Base: depth=12, n_est=1200, lr=0.06 ---
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.2, "min_child_weight": 3},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.4, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.2, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.2, "min_child_weight": 3},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.4, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.4, "min_child_weight": 3},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.2, "min_child_weight": 2},

    # --- Slightly higher learning rate (0.08) ---
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.2, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.4, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.4, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.2, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.4, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1200, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.4, "min_child_weight": 3},

    # --- Slightly longer training (n_est=1400) ---
    {"max_depth": 12, "n_estimators": 1400, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.2, "min_child_weight": 2},
    {"max_depth": 12, "n_estimators": 1400, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.4, "min_child_weight": 2},
]

# ------------------------------------------------------------
# Utility: Save confusion matrix plots
# ------------------------------------------------------------
def save_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# Data Loading & Preprocessing
# ============================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Split features and labels
X_full = train_df.drop(columns=["patient_id", "stage_clean"])
y_full = train_df["stage_clean"].values
X_test = test_df.drop(columns=["patient_id", "stage_clean"])
y_test_raw = test_df["stage_clean"].values

# Encode stage labels into integers
le = LabelEncoder()
y_full = le.fit_transform(y_full)
y_test = le.transform(y_test_raw)
classes = le.classes_

# Split training set into train/validation subsets
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=RS
)

# Standardize features
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply higher sample weight to Stage III samples
sample_weight = np.ones_like(y_tr, dtype=float)
stage3_idx = list(classes).index("Stage III")
sample_weight[y_tr == stage3_idx] *= 1.3

# ============================================================
# Manual Training Loop
# ============================================================
results = []
best_val_acc, best_model, best_params = -1.0, None, None
final_model_path = os.path.join(SAVE_DIR, "final_model.pkl")

for i, params in enumerate(param_list, 1):
    print(f"\n[{i}/{len(param_list)}] Training with params: {params}")
    model = XGBClassifier(
        objective="multi:softprob", num_class=len(classes),
        eval_metric="mlogloss", tree_method="hist",
        deterministic_histogram=True, random_state=RS, **params
    )

    # Train model
    t0 = time.time()
    model.fit(X_tr_scaled, y_tr, sample_weight=sample_weight, verbose=False)
    t = time.time() - t0

    # Evaluate on validation and test sets
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    acc_val = accuracy_score(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred, average="macro")
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average="macro")

    print(f"Val acc={acc_val:.3f}, Test acc={acc_test:.3f}")
    results.append({**params, "val_acc": acc_val, "test_acc": acc_test,
                    "f1_val": f1_val, "f1_test": f1_test, "train_time": round(t, 2)})

    # Save best-by-validation model
    if acc_val > best_val_acc:
        best_val_acc, best_model, best_params = acc_val, model, params
        print(f"New best model found (Val acc={acc_val:.3f})")

# ============================================================
# Save Results & Reports
# ============================================================
results_df = pd.DataFrame(results)
csv_path = os.path.join(SAVE_DIR, "manual_param_results15.csv")
results_df.to_csv(csv_path, index=False)

print(f"\nAll results saved → {csv_path}")
print(f"Best params: {best_params}")
print(f"Best validation acc: {best_val_acc:.3f}")

# Plot accuracy comparison bar chart
idx = np.arange(len(results_df))
width = 0.4
plt.figure(figsize=(12, 5))
plt.bar(idx - width/2, results_df["val_acc"].values, width, label="Val acc")
plt.bar(idx + width/2, results_df["test_acc"].values, width, label="Test acc")
plt.xticks(idx, [str(i+1) for i in range(len(results_df))])
plt.xlabel("Param set index (1–15)")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "acc_bar_compare.png"), dpi=300)
plt.close()

# ============================================================
# Final Evaluation & Model Export
# ============================================================
y_val_best = best_model.predict(X_val_scaled)
y_test_best = best_model.predict(X_test_scaled)

# Classification reports
val_report_df = pd.DataFrame(classification_report(
    y_val, y_val_best, target_names=classes, digits=3, output_dict=True
)).transpose()
test_report_df = pd.DataFrame(classification_report(
    y_test, y_test_best, target_names=classes, digits=3, output_dict=True
)).transpose()

print("\n=== Validation Report (Best-by-Val) ===")
print(val_report_df)
print("\n=== Test Report (Best-by-Val) ===")
print(test_report_df)

# Save reports and confusion matrices
val_report_df.to_csv(os.path.join(SAVE_DIR, "Val_Report_best.csv"))
test_report_df.to_csv(os.path.join(SAVE_DIR, "Test_Report_best.csv"))
save_confusion(y_val, y_val_best, classes,
               f"Validation acc={best_val_acc:.3f}",
               os.path.join(SAVE_DIR, "Val_CM_best.png"))
save_confusion(y_test, y_test_best, classes,
               f"Test acc={accuracy_score(y_test, y_test_best):.3f}",
               os.path.join(SAVE_DIR, "Test_CM_best.png"))

# Save full model bundle (for client use)
bundle = {
    "model": best_model,
    "scaler": scaler,
    "columns": X_full.columns.tolist(),
    "classes": classes.tolist()
}
joblib.dump(bundle, final_model_path)
print(f"\n✅ Final model bundle saved → {final_model_path}")
print("Done.")
