"""
============================================================
Federated Client Deployment (Flower)
This script loads the trained local model (final_model.pkl),
evaluates its performance on the test metadata, and connects
to the Flower server as a NumPyClient for federated prediction
fusion. It supports weighted uploading of modality-specific
probabilities to the central server.
============================================================
"""
import json
import numpy as np
import pandas as pd
import flwr as fl
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# -------------------- Configuration --------------------
SERVER = "192.168.0.16:8080"       # Flower server address
TEST_META = "test_metadata.csv"    # Metadata containing patient_id and label
DATA_PATH = "DNA_test.csv"         # DNA test feature file
MODEL_PATH = "final_model.pkl"     # Trained local model bundle
MODALITY = "DNA"                   # Modality name (for multi-modal fusion)
N_CLASSES = 4                      # Number of cancer stages
WEIGHT = 0.2                       # Fusion weight for this client

# -------------------- Load Metadata --------------------
meta = pd.read_csv(TEST_META)
le = LabelEncoder()
meta["label_id"] = le.fit_transform(meta["label"].astype(str))
pids = meta["patient_id"].astype(str).tolist()
y_true = meta["label_id"].tolist()

# -------------------- Load Test Features --------------------
df = pd.read_csv(DATA_PATH)
df = df[df["patient_id"].isin(pids)].copy().reset_index(drop=True)
X = df.drop(columns=["patient_id", "stage_clean"], errors="ignore").values
pid_order = df["patient_id"].tolist()

# -------------------- Load Model Bundle --------------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle.get("scaler", None)
train_columns = bundle["columns"]
classes = bundle.get("classes", [])
if len(classes) > 0:
    le = LabelEncoder()
    le.fit(classes)
else:
    le = LabelEncoder()

# Align feature columns and standardize
df = pd.read_csv(DATA_PATH)
df = df[df["patient_id"].isin(pids)].copy().reset_index(drop=True)
X = df.drop(columns=["patient_id", "stage_clean"], errors="ignore")
X = X.reindex(columns=train_columns, fill_value=0)
X = scaler.transform(X)
pid_order = df["patient_id"].tolist()

# -------------------- Local Evaluation --------------------
probs = model.predict_proba(X)
preds = np.argmax(probs, axis=1)
acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average="macro")
print(f"\n🔍 [Local Evaluation] Accuracy={acc:.4f}, Macro-F1={f1:.4f}\n")

# -------------------- Define Flower Client --------------------
class CNVClient(fl.client.NumPyClient):
    def __init__(self, model, X, pids, y_true, modality: str, weight: float):
        self.model = model
        self.X = X
        self.pids = pids
        self.y_true = np.array(y_true)
        self.modality = modality
        self.weight_for_fusion = float(weight)

    def get_parameters(self, config):
        # No local training in this phase
        return []

    def fit(self, parameters, config):
        # Skip training, only inference
        return [], 0, {}

    def evaluate(self, parameters, config):
        # Perform local inference and upload probabilities
        task = config.get("task", "")
        metrics = {}

        probs = self.model.predict_proba(self.X)
        preds = np.argmax(probs, axis=1)
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

        print(f"✅ [{self.modality}] Upload successful. {len(self.pids)} samples sent to server.")
        return 0.0, len(self.pids), metrics

# -------------------- Connect to Flower Server --------------------
client = CNVClient(model, X, pid_order, y_true, MODALITY, WEIGHT)
fl.client.start_client(server_address=SERVER, client=client.to_client())
