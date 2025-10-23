import json
import numpy as np
import pandas as pd
import flwr as fl
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ===== 参数 =====
SERVER     = "192.168.0.8:8080"   # 改成你的服务器IP
#SERVER = "127.0.0.1:8080"
TEST_META  = "test_metadata.csv"   # 测试集
DATA_PATH  = "DNA_test.csv"        # DNA特征文件
MODEL_PATH = "best_dna_model.pkl"  # 已训练好的模型
MODALITY   = "DNA"
N_CLASSES  = 4
WEIGHT     = 0.2

# ===== 读取测试集 (带标签) =====
meta = pd.read_csv(TEST_META)
assert "patient_id" in meta.columns and "label" in meta.columns

# 标签编码
le = LabelEncoder()
meta["label_id"] = le.fit_transform(meta["label"].astype(str))

pids = meta["patient_id"].astype(str).tolist()
y_true = meta["label_id"].tolist()


# ===== 加载 DNA 特征数据 =====
df = pd.read_csv(DATA_PATH)
df = df[df["patient_id"].isin(pids)].copy().reset_index(drop=True)
X = df.drop(columns=["patient_id", "stage_clean"], errors="ignore").values
pid_order = df["patient_id"].tolist()


# ===== 加载模型和预处理器 =====
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle.get("scaler", None)
train_columns = bundle["columns"]

# 类别标签信息（如果有）
classes = bundle.get("classes", [])
if len(classes) > 0:
    le = LabelEncoder()
    le.fit(classes)
else:
    le = LabelEncoder()

# 对齐特征列
df = pd.read_csv(DATA_PATH)
df = df[df["patient_id"].isin(pids)].copy().reset_index(drop=True)
X = df.drop(columns=["patient_id", "stage_clean"], errors="ignore")

# 确保和训练时列一致
X = X.reindex(columns=train_columns, fill_value=0)
X = scaler.transform(X)   # 标准化
pid_order = df["patient_id"].tolist()

# ===== 本地评估 =====
probs = model.predict_proba(X)
preds = np.argmax(probs, axis=1)

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average="macro")
print(f"\n🔍 [Local Evaluation] Accuracy={acc:.4f}, Macro-F1={f1:.4f}\n")

# ===== Flower 客户端 =====
class CNVClient(fl.client.NumPyClient):
    def __init__(self, model, X, pids, y_true, modality: str, weight: float):
        self.model = model
        self.X = X
        self.pids = pids
        self.y_true = np.array(y_true)
        self.modality = modality
        self.weight_for_fusion = float(weight)

    def get_parameters(self, config):
        return []  # 不训练，返回空

    def fit(self, parameters, config):
        return [], 0, {}  # 不训练

    def evaluate(self, parameters, config):
        task = config.get("task", "")
        metrics = {}

        # ===== 预测 =====
        probs = self.model.predict_proba(self.X)
        preds = np.argmax(probs, axis=1)

        # ===== 本地指标（再次打印） =====
        acc = accuracy_score(self.y_true, preds)
        f1 = f1_score(self.y_true, preds, average="macro")
        print(f"[Flower Evaluation] Accuracy={acc:.4f}, Macro-F1={f1:.4f}")

        if task == "predict":
            rows = []
            for pid, p in zip(self.pids, probs):
                p = np.clip(p, 1e-9, 1.0)
                p = p / p.sum()
                rows.append({
                    "patient_id": pid,
                    "probs": p.tolist(),
                    "modality": self.modality,
                    "weight": self.weight_for_fusion,
                })
            metrics = {
                "preds_json": json.dumps(rows).encode("utf-8"),
                "acc_local": float(acc),
                "f1_local": float(f1),
            }

        return 0.0, len(self.pids), metrics


# ===== open the client =====
client = CNVClient(model, X, pid_order, y_true, MODALITY, WEIGHT)
fl.client.start_client(
    server_address=SERVER,
    client=client.to_client(),
)



