# select_top1000_features.py
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

# ========== Config ==========
DATA_PATH = "BRCA_DNA(CNV)_stage.csv"   # Input CNV+stage file
TOP_K = 1000                            # Number of features to select
OUT_PATH = "DNA-1000.csv"       # Output file
SEED = 42
# ============================

np.random.seed(SEED)

# 1) Load data
df = pd.read_csv(DATA_PATH)
assert "patient_id" in df.columns and "stage_clean" in df.columns, "Input must include patient_id and stage_clean"
X_df = df.drop(columns=["patient_id", "stage_clean"])
genes_all = list(X_df.columns)
y_raw = df["stage_clean"].values

# Encode stage labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Classes:", le.classes_)

# 2) Remove constant / near-constant features
vt = VarianceThreshold(threshold=1e-6)
X_vt = vt.fit_transform(X_df)
genes_vt = np.array(genes_all)[vt.get_support()]
print(f"Remaining features after variance filter: {len(genes_vt)}")

# 3) ANOVA F-test to select top-K features
k = min(TOP_K, len(genes_vt))
skb = SelectKBest(score_func=f_classif, k=k)
X_sel = skb.fit_transform(X_vt, y)
genes_sel = genes_vt[skb.get_support()]

print(f"Selected top {len(genes_sel)} features")

# 4) Save filtered dataset
X_top = pd.concat([df[["patient_id", "stage_clean"]], X_df[genes_sel]], axis=1)
X_top.to_csv(OUT_PATH, index=False)

print(f"Saved filtered dataset to {OUT_PATH}")
