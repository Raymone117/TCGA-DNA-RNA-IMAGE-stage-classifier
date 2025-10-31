import pandas as pd

DATA_PATH = "DNA-1000.csv"

# 固定测试集 ID 顺序（不要打乱）
overlap_ids = [
    "TCGA-A2-A3XZ",
    "TCGA-A7-A426",
    "TCGA-A2-A04N",
    "TCGA-A8-A09X",
    "TCGA-PL-A8LX",
    "TCGA-E2-A1II",
    "TCGA-E2-A14U",
    "TCGA-A2-A0T3",
    "TCGA-A2-A04R",
    "TCGA-E9-A54Y",
    "TCGA-A8-A09Q",
    "TCGA-A8-A086",
    "TCGA-E9-A1NC",
    "TCGA-E9-A1ND",
    "TCGA-S3-A6ZF",
    "TCGA-WT-AB41",
    "TCGA-A1-A0SN",
    "TCGA-A2-A0YM",
    "TCGA-A8-A06U",
    "TCGA-A8-A07F",
    "TCGA-A8-A09K",
    "TCGA-A8-A082",
    "TCGA-AC-A2B8",
    "TCGA-AR-A24H",
    "TCGA-B6-A0WZ",
    "TCGA-A2-A0T1",
    "TCGA-A8-A06T",
    "TCGA-A8-A07L"
]

# 读取完整数据
df = pd.read_csv(DATA_PATH)
df["patient_id"] = df["patient_id"].astype(str)

# === 保证 test.csv 的顺序与 overlap_ids 完全一致 ===
df_test = df.set_index("patient_id").loc[overlap_ids].reset_index()

# 训练集是剩下的
df_train = df[~df["patient_id"].isin(overlap_ids)].copy()

# 保存结果（顺序不乱）
df_train.to_csv("DNA_train.csv", index=False)
df_test.to_csv("DNA_test.csv", index=False)

print("✅ Done!")
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
print("📋 Test order confirmed:", df_test["patient_id"].tolist()[:5], "...")
