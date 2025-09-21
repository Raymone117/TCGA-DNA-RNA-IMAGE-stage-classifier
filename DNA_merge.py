import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

CNV_DIR = "TCGA_BRCA_CNV"
METADATA_PATH = "metadata.json"
CLINICAL_PATH = "clinical.tsv"
OUTPUT_PATH = "BRCA_DNA(CNV)_stage.csv"


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


stage_order = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
clinical_df["stage_num"] = clinical_df["stage_clean"].map(stage_order)
clinical_df = clinical_df.sort_values("stage_num", ascending=False).drop_duplicates("patient_id")

print(f"clinical.tsv processed, {len(clinical_df)} patients have valid stage labels (after keeping latest stage)")


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

    df_gene = df.groupby("gene_name")["copy_number"].median().reset_index()
    cnv_dict = dict(zip(df_gene["gene_name"], df_gene["copy_number"]))
    cnv_data[patient_id] = cnv_dict

print(f"Collected CNV data for {len(cnv_data)} patients")

cnv_df = pd.DataFrame.from_dict(cnv_data, orient="index")
cnv_df.index.name = "patient_id"
cnv_df.reset_index(inplace=True)

merged = pd.merge(clinical_df[["patient_id", "stage_clean"]],
                  cnv_df, on="patient_id", how="inner")

merged = merged.fillna(0)

print(f"Merge completed, final dataset shape: {merged.shape}")


merged.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")

print("Data Summary:")
print(f"Total patients merged: {merged['patient_id'].nunique()}")
print("Patients per stage:")
print(merged['stage_clean'].value_counts())
print(f"Total gene features: {merged.shape[1]-2}")

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
