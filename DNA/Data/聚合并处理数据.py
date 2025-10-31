import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Step 1: 配置路径
CNV_DIR = "TCGA_BRCA_CNV"           # 存放 CNV 文件的文件夹
METADATA_PATH = "metadata.json"     # GDC 下载的 metadata.json
CLINICAL_PATH = "clinical.tsv"      # GDC 下载的 clinical.tsv
OUTPUT_PATH = "BRCA_DNA(CNV)_stage.csv"  # 输出结果

# Step 2: 处理 metadata.json (文件名 -> patient_id)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

file_to_patient = {}
for entry in metadata:
    file_name = entry.get("file_name")
    entities = entry.get("associated_entities", [])
    if not entities:
        continue
    entity_id = entities[0]["entity_submitter_id"]
    patient_id = "-".join(entity_id.split("-")[:3])  # 保留 TCGA-XX-XXXX
    file_to_patient[file_name] = patient_id

print(f"metadata.json processed, got {len(file_to_patient)} file-to-patient mappings")

# Step 3: 处理 clinical.tsv
clinical = pd.read_csv(CLINICAL_PATH, sep="\t")

if "diagnoses.ajcc_pathologic_stage" not in clinical.columns:
    raise ValueError("'diagnoses.ajcc_pathologic_stage' not found in clinical.tsv")

clinical_df = clinical[["cases.submitter_id", "diagnoses.ajcc_pathologic_stage"]].dropna()
clinical_df = clinical_df.rename(columns={
    "cases.submitter_id": "patient_id",
    "diagnoses.ajcc_pathologic_stage": "stage"
})

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

# 保留最晚期
stage_order = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
clinical_df["stage_num"] = clinical_df["stage_clean"].map(stage_order)
clinical_df = clinical_df.sort_values("stage_num", ascending=False).drop_duplicates("patient_id")

print(f"clinical.tsv processed, {len(clinical_df)} patients have valid stage labels (after keeping latest stage)")

# Step 4: 处理 CNV 文件
cnv_data = {}

for file_name, patient_id in tqdm(file_to_patient.items()):
    file_path = os.path.join(CNV_DIR, file_name)
    if not os.path.exists(file_path):
        continue

    try:
        df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        print(f"Cannot read {file_name}: {e}")
        continue

    if "gene_name" not in df.columns or "copy_number" not in df.columns:
        continue

    # 对同一个 gene_name 取中位数，避免重复
    df_gene = df.groupby("gene_name")["copy_number"].median().reset_index()
    cnv_dict = dict(zip(df_gene["gene_name"], df_gene["copy_number"]))
    cnv_data[patient_id] = cnv_dict

print(f"Collected CNV data for {len(cnv_data)} patients")

# Step 5: 转换为 DataFrame 并合并临床信息
cnv_df = pd.DataFrame.from_dict(cnv_data, orient="index")
cnv_df.index.name = "patient_id"
cnv_df.reset_index(inplace=True)

merged = pd.merge(clinical_df[["patient_id", "stage_clean"]],
                  cnv_df, on="patient_id", how="inner")

# 填补缺失值为 0（中性 copy number）
merged = merged.fillna(0)

print(f"Merge completed, final dataset shape: {merged.shape}")

# Step 6: 保存结果
merged.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")

# Step 7: 汇总信息
print("Data Summary:")
print(f"Total patients merged: {merged['patient_id'].nunique()}")
print("Patients per stage:")
print(merged['stage_clean'].value_counts())
print(f"Total gene features: {merged.shape[1]-2}")

# Step 8: 可视化分期分布
plt.figure(figsize=(6, 4))
stage_counts = merged['stage_clean'].value_counts().sort_index()
stage_counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Patient Distribution by Stage (BRCA CNV)")
plt.xlabel("Stage")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("stage_distribution.png", dpi=300)

print("Stage distribution plot saved as stage_distribution.png")
