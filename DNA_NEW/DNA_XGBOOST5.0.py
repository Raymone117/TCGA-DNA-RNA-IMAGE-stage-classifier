# ============================================================
# DNA_XGBOOST 5.0
# RAYMONE
# ============================================================

import os, joblib, time, warnings, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------ Config ------------------
RS = 192
random.seed(RS)
np.random.seed(RS)
os.environ["PYTHONHASHSEED"] = str(RS)

TRAIN_PATH = "DNA_train.csv"
TEST_PATH  = "DNA_test.csv"
SAVE_DIR   = "models_5.0_final"
os.makedirs(SAVE_DIR, exist_ok=True)

best_params = {
    "max_depth": 12,
    "n_estimators": 1200,
    "learning_rate": 0.06,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.2,
    "min_child_weight": 2
}
stage3_weight = 1.3

# ------------------ Utility ------------------
def save_confusion(y_true, y_pred, labels, title, path):
    """Save confusion matrix as image."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def make_model(params, num_classes):
    """Create deterministic XGBoost model."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        deterministic_histogram=True,
        random_state=RS,
        **params
    )

# ------------------ Load Data ------------------
print("📂 Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_full = train_df.drop(columns=["patient_id", "stage_clean"])
y_full = train_df["stage_clean"].values
X_test = test_df.drop(columns=["patient_id", "stage_clean"])
y_test_raw = test_df["stage_clean"].values

le = LabelEncoder()
y_full = le.fit_transform(y_full)
y_test = le.transform(y_test_raw)
classes = le.classes_
num_classes = len(classes)

# Split (fixed for reproducibility)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=RS
)

# ------------------ Preprocessing ------------------
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ------------------ Training ------------------
sample_weight = np.ones_like(y_tr, dtype=float)
stage3_idx = list(classes).index("Stage III")
sample_weight[y_tr == stage3_idx] *= stage3_weight

model = make_model(best_params, num_classes)
print("🚀 Training reproducible best model...")
t0 = time.time()
model.fit(X_tr_scaled, y_tr, sample_weight=sample_weight, verbose=False)
print(f"✅ Done in {time.time()-t0:.1f}s")

# ------------------ Evaluate ------------------
# Validation
y_val_pred = model.predict(X_val_scaled)
acc_val = accuracy_score(y_val, y_val_pred)
f1_val = f1_score(y_val, y_val_pred, average="macro")

# Test
y_test_pred = model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred, average="macro")

print(f"\n📊 Validation: acc={acc_val:.3f}, F1={f1_val:.3f}")
print(f"🏁 Test: acc={acc_test:.3f}, F1={f1_test:.3f}")

# ------------------ Reports ------------------
print("\n=== Test Report ===")
report_dict = classification_report(y_test, y_test_pred, target_names=classes, digits=3, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_csv = os.path.join(SAVE_DIR, f"DNA_Report_acc{acc_test:.3f}.csv")
report_df.to_csv(report_csv, index=True)
print(report_df)
print(f"\n💾 Classification report saved to: {report_csv}")

# ------------------ Confusion Matrices ------------------
val_cm_path = os.path.join(SAVE_DIR, f"DNA_Val_CM_acc{acc_val:.3f}.png")
test_cm_path = os.path.join(SAVE_DIR, f"DNA_Test_CM_acc{acc_test:.3f}.png")

save_confusion(y_val, y_val_pred, classes, f"Validation acc={acc_val:.3f}", val_cm_path)
save_confusion(y_test, y_test_pred, classes, f"Test acc={acc_test:.3f}", test_cm_path)

print(f"🧩 Confusion matrices saved:\n - Validation → {val_cm_path}\n - Test → {test_cm_path}")

# ------------------ Save Model ------------------
model_path = os.path.join(SAVE_DIR, f"DNA_Model5.0_acc{acc_test:.3f}.pkl")
joblib.dump({
    "model": model,
    "scaler": scaler,
    "classes": classes.tolist(),
    "params": best_params,
    "stage3_w": stage3_weight,
    "acc_val": acc_val,
    "acc_test": acc_test
}, model_path)

print(f"\n💾 Model saved to: {model_path}")
print("✅ Reproducible retrain complete.")
