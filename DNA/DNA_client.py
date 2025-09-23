import json
import numpy as np
import pandas as pd
import flwr as fl
import joblib

# ===== 参数 =====
SERVER     = "127.0.0.1:8080"    # 如果服务器在本机
TEST_META  = "test_metadata.csv" # 测试集，含 patient_id,label
DATA_PATH  = "DNA-1000.csv"      # 训练用的特征文件
MODEL_PATH = "cnv_xgb_best.pkl"  # 训练好的 XGBoost 模型
MODALITY   = "DNA"
N_CLASSES  = 4
WEIGHT     = 1.0

# ===== 加载测试病人列表 =====
meta = pd.read_csv(TEST_META)
assert "patient_id" in meta.columns
pids = meta["patient_id"].astype(str).tolist()

# ===== 加载 CNV 特征数据集 =====
df = pd.read_csv(DATA_PATH)
df = df[df["patient_id"].isin(pids)].copy().reset_index(drop=True)
X = df.drop(columns=["patient_id", "stage_clean"], errors="ignore").values
pid_order = df["patient_id"].tolist()

# ===== 加载已训练好的 XGBoost 模型 =====
model = joblib.load(MODEL_PATH)

# ===== Flower 客户端 =====
class CNVClient(fl.client.NumPyClient):
    def __init__(self, model, X, pids, modality: str, weight: float):
        self.model = model
        self.X = X
        self.pids = pids
        self.modality = modality
        self.weight_for_fusion = float(weight)

    def get_parameters(self, config):
        return []  # 不训练，返回空

    def fit(self, parameters, config):
        return [], 0, {}

    def evaluate(self, parameters, config):
        task = config.get("task", "")
        metrics = {}
        if task == "predict":
            probs = self.model.predict_proba(self.X)
            rows = []
            for pid, p in zip(self.pids, probs):
                p = np.clip(p, 1e-9, 1.0)  # 避免概率为 0
                p = p / p.sum()
                rows.append({
                    "patient_id": pid,
                    "probs": p.tolist(),
                    "modality": self.modality,
                    "weight": self.weight_for_fusion,
                })
            metrics = {"preds_json": json.dumps(rows).encode("utf-8")}
        return 0.0, len(self.pids), metrics

# ===== 启动客户端 =====
client = CNVClient(model, X, pid_order, MODALITY, WEIGHT)
fl.client.start_numpy_client(server_address=SERVER, client=client)
