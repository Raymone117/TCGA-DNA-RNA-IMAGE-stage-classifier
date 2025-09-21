import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from itertools import product
import joblib

# ------------------ config ------------------
SEED = 42
DATA_PATH = "DNA-1000.csv"
TEST_SIZE = 0.2   # 80/20 split
# --------------------------------------------

# 1) Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["patient_id", "stage_clean"])
y_raw = df["stage_clean"].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
num_classes = len(classes)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# ------------------ Training + Grid Search ------------------
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.1, 0.05, 0.01],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

search_space = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["learning_rate"],
    param_grid["subsample"],
    param_grid["colsample_bytree"]
))

best_f1, best_model, best_params = -1, None, None
for n_estimators, max_depth, lr, subsample, colsample in search_space:
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": lr,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "objective": "multi:softmax",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "random_state": SEED
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")

    print(f"Params {params} → Macro F1 = {f1:.4f}")

    if f1 > best_f1:
        best_f1, best_model, best_params = f1, model, params

print("\n✅ Best params:", best_params, "with Macro F1 =", best_f1)

# ------------------ Final Evaluation ------------------
y_pred = best_model.predict(X_val)

report = classification_report(y_val, y_pred, target_names=classes, digits=3, output_dict=True)
print("\n=== Final Validation Report ===")
for cls in classes:
    precision = report[cls]["precision"]
    recall    = report[cls]["recall"]
    f1        = report[cls]["f1-score"]
    support   = report[cls]["support"]
    print(f"{cls:>8} | Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, N={support}")

print(f"\nAccuracy    = {report['accuracy']:.3f}")
print(f"Macro Avg   = P:{report['macro avg']['precision']:.3f}, "
      f"R:{report['macro avg']['recall']:.3f}, "
      f"F1:{report['macro avg']['f1-score']:.3f}")
print(f"Weighted Avg= P:{report['weighted avg']['precision']:.3f}, "
      f"R:{report['weighted avg']['recall']:.3f}, "
      f"F1:{report['weighted avg']['f1-score']:.3f}")

cm = confusion_matrix(y_val, y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Best XGBoost Model (80/20 split)")
plt.show()

# ------------------ Save best model ------------------
MODEL_PATH = "cnv_xgb_best.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"✅ Best XGBoost model saved to {MODEL_PATH}")
