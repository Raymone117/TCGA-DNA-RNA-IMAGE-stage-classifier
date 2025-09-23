import numpy as np
import pandas as pd
import anndata as ad
import joblib
import json
from scipy.sparse import issparse
from sklearn.metrics import classification_report, confusion_matrix
import flwr as fl

# ===== 参数设置 =====
TEST_H5AD_PATH  = "/Users/xin/Desktop/DATA5703/TCGA-DNA-RNA-IMAGE-stage-classifier/RNA/RNA_test.h5ad"
SELECTOR_PATH   = "/Users/xin/Desktop/DATA5703/TCGA-DNA-RNA-IMAGE-stage-classifier/RNA//RNA_selector.pkl"
MODEL_PATH      = "/Users/xin/Desktop/DATA5703/TCGA-DNA-RNA-IMAGE-stage-classifier/RNA//RNA_model.pkl"
SERVER_ADDRESS  = "127.0.0.1:8080"
MODALITY        = "RNA"
WEIGHT          = 1.0

label_map = {"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3}
label_names = list(label_map.keys())

class RNAClient(fl.client.NumPyClient):
    def __init__(self, test_h5ad_path, selector_path, model_path, modality, weight):
        self.modality = modality
        self.weight = weight
        self.rows = []
        self._load_and_predict(test_h5ad_path, selector_path, model_path)

    def _load_and_predict(self, h5ad_path, selector_path, model_path):
        # === 1. 加载数据 ===
        adata = ad.read_h5ad(h5ad_path)
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        y_raw = adata.obs["stage"].values
        pids = adata.obs["patient_id"].astype(str).values

        y_true = np.array([label_map.get(s, 3) for s in y_raw])

        # === 2. 加载 selector 和模型 ===
        selector = joblib.load(selector_path)
        model = joblib.load(model_path)

        # === 3. 特征选择 + 预测 ===
        X_sel = selector.transform(X)
        y_pred = model.predict(X_sel)
        y_prob = model.predict_proba(X_sel)

        # === 4. 输出评估结果 ===
        print("📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=label_names))

        print("📊 Confusion Matrix:")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred), index=label_names, columns=label_names))

        # === 5. 封装为 JSON 结构 ===
        for i, prob in enumerate(y_prob):
            self.rows.append({
                "patient_id": pids[i],
                "probs": prob.tolist(),
                "modality": self.modality,
                "weight": self.weight
            })

        print(f"✅ 已生成 {len(self.rows)} 条预测")

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        return [], 0, {}

    def evaluate(self, parameters, config):
        task = config.get("task", "")
        metrics = {}
        if task == "predict":
            print(f"📤 RNA 客户端上传 {len(self.rows)} 条预测")
            metrics = {
                "preds_json": json.dumps(self.rows).encode("utf-8")
            }
        return 0.0, len(self.rows), metrics

# ===== 启动客户端 =====
client = RNAClient(TEST_H5AD_PATH, SELECTOR_PATH, MODEL_PATH, MODALITY, WEIGHT)
fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)