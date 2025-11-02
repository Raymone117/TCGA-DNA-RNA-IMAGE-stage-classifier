"""
============================================================
Data Preprocess (DNA CNV)
This script constructs the full DNA CNV dataset by merging GDC metadata,
clinical staging records, and gene-level CNV files. It then performs feature
filtering, fixed train/test splitting, and visualization (PCA/UMAP/distribution).
============================================================
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
try:
    import umap.umap_ as umap
except ImportError:
    import umap

# Input directories and parameters
CNV_DIR = "TCGA_BRCA_CNV"         # Folder containing TCGA CNV segment files
METADATA_PATH = "metadata.json"   # Metadata linking CNV files to patient IDs
CLINICAL_PATH = "clinical.tsv"    # Clinical file containing stage information
TOP_K = 1000                      # Number of top genes to retain via ANOVA F-test
SEED = 42
np.random.seed(SEED)

# Output paths
OUT_FULL = "BRCA_DNA(CNV)_stage.csv"
OUT_TRAIN = "DNA_train.csv"
OUT_TEST = "DNA_test.csv"
SAVE_DIR = "report_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Fixed 28-patient test set for reproducibility
TEST_IDS = [
    "TCGA-A2-A3XZ", "TCGA-A7-A426", "TCGA-A2-A04N", "TCGA-A8-A09X",
    "TCGA-PL-A8LX", "TCGA-E2-A1II", "TCGA-E2-A14U", "TCGA-A2-A0T3",
    "TCGA-A2-A04R", "TCGA-E9-A54Y", "TCGA-A8-A09Q", "TCGA-A8-A086",
    "TCGA-E9-A1NC", "TCGA-E9-A1ND", "TCGA-S3-A6ZF", "TCGA-WT-AB41",
    "TCGA-A1-A0SN", "TCGA-A2-A0YM", "TCGA-A8-A06U", "TCGA-A8-A07F",
    "TCGA-A8-A09K", "TCGA-A8-A082", "TCGA-AC-A2B8", "TCGA-AR-A24H",
    "TCGA-B6-A0WZ", "TCGA-A2-A0T1", "TCGA-A8-A06T", "TCGA-A8-A07L"
]


# ============================================================
# Step 1. Parse metadata and clinical files
# ============================================================
# Build a mapping from CNV file names to patient IDs using metadata.json
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

file_to_patient = {}
for entry in metadata:
    name = entry.get("file_name")
    ent = entry.get("associated_entities", [])
    if not ent:
        continue
    pid = "-".join(ent[0]["entity_submitter_id"].split("-")[:3])
    file_to_patient[name] = pid

print(f"✅ Loaded metadata: {len(file_to_patient)} file-patient mappings")

# Load clinical file and extract stage information
clinical = pd.read_csv(CLINICAL_PATH, sep="\t")
if "diagnoses.ajcc_pathologic_stage" not in clinical.columns:
    raise ValueError("Missing column 'diagnoses.ajcc_pathologic_stage' in clinical.tsv")

clinical_df = clinical[["cases.submitter_id", "diagnoses.ajcc_pathologic_stage"]].dropna()
clinical_df.columns = ["patient_id", "stage"]

# Normalize stage labels into Stage I–IV
def clean_stage(x):
    if not isinstance(x, str):
        return None
    x = x.upper()
    if "I" in x and not "II" in x and not "III" in x and not "IV" in x:
        return "Stage I"
    elif "II" in x and not "III" in x and not "IV" in x:
        return "Stage II"
    elif "III" in x and not "IV" in x:
        return "Stage III"
    elif "IV" in x:
        return "Stage IV"
    else:
        return None

clinical_df["stage_clean"] = clinical_df["stage"].apply(clean_stage)
clinical_df = clinical_df.dropna(subset=["stage_clean"])

# Keep the most advanced stage per patient
order = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
clinical_df["stage_num"] = clinical_df["stage_clean"].map(order)
clinical_df = clinical_df.sort_values("stage_num", ascending=False).drop_duplicates("patient_id")

print(f"✅ Processed clinical.tsv: {len(clinical_df)} patients with valid stages")


# ============================================================
# Step 2. Extract CNV data
# ============================================================
# Read all CNV files and aggregate gene-level copy number per patient
cnv_data = {}
for file_name, pid in tqdm(file_to_patient.items(), desc="Processing CNV files"):
    path = os.path.join(CNV_DIR, file_name)
    if not os.path.exists(path):
        continue
    try:
        df = pd.read_csv(path, sep="\t")
    except:
        continue
    if "gene_name" not in df.columns or "copy_number" not in df.columns:
        continue
    df_gene = df.groupby("gene_name")["copy_number"].median().reset_index()
    cnv_data[pid] = dict(zip(df_gene["gene_name"], df_gene["copy_number"]))

# Combine all CNV data into a single DataFrame and merge with stage labels
cnv_df = pd.DataFrame.from_dict(cnv_data, orient="index").reset_index().rename(columns={"index": "patient_id"})
merged = pd.merge(clinical_df[["patient_id", "stage_clean"]], cnv_df, on="patient_id", how="inner").fillna(0)
merged.to_csv(OUT_FULL, index=False)
print(f"✅ Full dataset saved: {OUT_FULL} (shape={merged.shape})")


# ============================================================
# Step 3. Dataset Splitting (then Feature Selection)
# ============================================================
print("\n🔹 Splitting dataset first, then performing feature selection ...")

df = merged.copy()
df["patient_id"] = df["patient_id"].astype(str)

# ---------- Split into Train/Test ----------
df_test = df.set_index("patient_id").loc[TEST_IDS].reset_index()
df_train = df[~df["patient_id"].isin(TEST_IDS)].copy()

# Extract features/labels for training
X_train = df_train.drop(columns=["patient_id", "stage_clean"])
y_train_raw = df_train["stage_clean"].values
X_test = df_test.drop(columns=["patient_id", "stage_clean"])
y_test_raw = df_test["stage_clean"].values

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

# ---------- Feature Selection on Training Only ----------
print("🔹 Selecting informative features from training data ...")
genes_all = list(X_train.columns)

# Variance filter
vt = VarianceThreshold(threshold=1e-6)
X_train_vt = vt.fit_transform(X_train)
genes_vt = np.array(genes_all)[vt.get_support()]

# ANOVA F-test
skb = SelectKBest(score_func=f_classif, k=min(TOP_K, len(genes_vt)))
X_train_sel = skb.fit_transform(X_train_vt, y_train)
genes_sel = genes_vt[skb.get_support()]

# ---------- Apply same selection to Test Set ----------
X_train_final = pd.DataFrame(X_train[genes_sel])
X_test_final = pd.DataFrame(X_test.reindex(columns=genes_sel, fill_value=0))

# ---------- Rebuild filtered DataFrames ----------
df_train_final = pd.concat([df_train[["patient_id", "stage_clean"]].reset_index(drop=True),
                            X_train_final.reset_index(drop=True)], axis=1)
df_test_final = pd.concat([df_test[["patient_id", "stage_clean"]].reset_index(drop=True),
                           X_test_final.reset_index(drop=True)], axis=1)

# ---------- Save Results ----------
df_train_final.to_csv(OUT_TRAIN, index=False)
df_test_final.to_csv(OUT_TEST, index=False)
print(f"✅ Train/Test feature selection completed → Train={df_train_final.shape}, Test={df_test_final.shape}")

# ============================================================
# Step 4. Visualization (PCA, UMAP, Stage Distribution)
# ============================================================
print("\n🎨 Generating visualizations (PCA + UMAP + Stage counts)")

# ---------- Prepare data ----------
# Combine train & test for visualization, but fit scaler/PCA/UMAP on training only
df_vis = pd.concat([df_train_final, df_test_final], axis=0).reset_index(drop=True)

X_train_vis = df_train_final.drop(columns=["patient_id", "stage_clean"])
X_all_vis = df_vis.drop(columns=["patient_id", "stage_clean"])
y_all_vis = df_vis["stage_clean"]

# Standardization (fit on training set)
scaler = StandardScaler()
X_scaled_all = scaler.fit(X_train_vis).transform(X_all_vis)

# ---------- PCA ----------
pca = PCA(n_components=2, random_state=SEED)
pca_result = pca.fit(X_train_vis).transform(X_scaled_all)
df_vis["PC1"], df_vis["PC2"] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(6, 5))
for s in sorted(df_vis["stage_clean"].unique()):
    subset = df_vis[df_vis["stage_clean"] == s]
    plt.scatter(subset["PC1"], subset["PC2"], label=s, s=10)
plt.legend()
plt.title("PCA of DNA CNV by Stage")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "DNA_PCA_by_Stage.png"), dpi=300)
plt.close()

# ---------- UMAP ----------
reducer = umap.UMAP(random_state=SEED)
umap_result = reducer.fit(X_train_vis).transform(X_scaled_all)
df_vis["UMAP1"], df_vis["UMAP2"] = umap_result[:, 0], umap_result[:, 1]

plt.figure(figsize=(6, 5))
for s in sorted(df_vis["stage_clean"].unique()):
    subset = df_vis[df_vis["stage_clean"] == s]
    plt.scatter(subset["UMAP1"], subset["UMAP2"], label=s, s=10)
plt.legend()
plt.title("UMAP of DNA CNV by Stage")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "DNA_UMAP_by_Stage.png"), dpi=300)
plt.close()

# ---------- Stage Distribution ----------
counts = y_all_vis.value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values, edgecolor="black")
plt.title("Sample Count per Stage (DNA CNV)")
plt.xlabel("Stage")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "DNA_Stage_Distribution.png"), dpi=300)
plt.close()

print("\n✅ All figures & datasets generated successfully.")
