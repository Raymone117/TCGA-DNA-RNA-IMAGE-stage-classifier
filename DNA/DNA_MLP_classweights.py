import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from itertools import product
import matplotlib.pyplot as plt

# ------------------ config ------------------
SEED = 42
EPOCHS = 50
PATIENCE = 5        # early stopping patience
DATA_PATH = "DNA-1000.csv"
# --------------------------------------------

# Reproducibility
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# 1) Load dataset
df = pd.read_csv(DATA_PATH)
X_full = df.drop(columns=["patient_id", "stage_clean"])
y_raw = df["stage_clean"].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
num_classes = len(classes)

# Train/val split (80/20)
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=SEED
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_val   = scaler.transform(X_val_df)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ MLP model ------------------
class CNVMLP(nn.Module):
    def __init__(self, in_dim, hid=256, dropout=0.3, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, hid),   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, num_classes)
        )
    def forward(self, x): return self.net(x)

# ------------------ Training + Eval ------------------
def train_eval(hparams):
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    hid = hparams["hid"]
    dropout = hparams["dropout"]

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size, shuffle=False
    )

    # Model
    model = CNVMLP(in_dim=X_train.shape[1], hid=hid, dropout=dropout, num_classes=num_classes).to(device)

    # Class weights
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1, best_ep = -1, 0
    patience_counter = 0
    best_state = None

    # Training loop
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()

        # Validation
        model.eval(); preds=[]; labels=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds += out.argmax(1).cpu().tolist()
                labels += yb.cpu().tolist()

        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_ep = ep
            best_state = model.state_dict()  # ✅ 保存最佳参数
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_f1, best_state

# ------------------ Grid Search ------------------
param_grid = {
    "lr": [1e-3, 5e-4, 1e-4],
    "hid": [128, 256, 512],
    "dropout": [0.3, 0.4, 0.5],
    "batch_size": [32, 64, 128],
}

search_space = list(product(param_grid["lr"], param_grid["hid"], param_grid["dropout"], param_grid["batch_size"]))

best_score, best_params, best_state = -1, None, None
for lr, hid, dropout, batch_size in search_space:
    hparams = {"lr": lr, "hid": hid, "dropout": dropout, "batch_size": batch_size}
    f1, state = train_eval(hparams)
    print(f"Params {hparams} → Best Macro F1 = {f1:.4f}")
    if f1 > best_score:
        best_score, best_params, best_state = f1, hparams, state

print("\n✅ Best params:", best_params, "with Macro F1 =", best_score)

# ------------------ Final Evaluation on Validation ------------------
# Rebuild best model and load weights
best_model = CNVMLP(in_dim=X_train.shape[1], hid=best_params["hid"],
                    dropout=best_params["dropout"], num_classes=num_classes).to(device)
best_model.load_state_dict(best_state)

best_model.eval(); preds=[]; labels=[]
with torch.no_grad():
    out = best_model(X_val_t.to(device))
    preds = out.argmax(1).cpu().tolist()
    labels = y_val_t.cpu().tolist()

report = classification_report(labels, preds, target_names=classes, output_dict=True, digits=3)
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

cm = confusion_matrix(labels, preds)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Best MLP Model (80/20 split)")
plt.show()

# Save the best model
MODEL_PATH = "cnv_mlp_best.pth"
torch.save(best_model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
