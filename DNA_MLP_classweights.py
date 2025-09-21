import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

# ------------------ config ------------------
SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DATA_PATH = "DNA-1000.csv"   # Already feature-selected dataset
# --------------------------------------------

# Reproducibility
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# 1) Load dataset
df = pd.read_csv(DATA_PATH)
X_full = df.drop(columns=["patient_id", "stage_clean"])
y_raw = df["stage_clean"].values
pids = df["patient_id"].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
num_classes = len(classes)
print("Classes:", classes)

# 2) Train/val split (70% / 30%)
X_train_df, X_val_df, y_train, y_val, pid_train, pid_val = train_test_split(
    X_full, y, pids, test_size=0.3, stratify=y, random_state=SEED
)

# 3) Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_val   = scaler.transform(X_val_df)

# 4) Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

# 5) Define MLP model
class CNVMLP(nn.Module):
    def __init__(self, in_dim, hid=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, hid),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, num_classes)
        )
    def forward(self, x): return self.net(x)

model = CNVMLP(in_dim=X_train.shape[1], hid=256, num_classes=num_classes).to(device)

# 6) Class weights to handle imbalance
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1e-6)   # inverse frequency
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 7) Training loop
for ep in range(1, EPOCHS+1):
    model.train(); tr_loss=0.0
    for xb,yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
        tr_loss += loss.item()
    # Validation
    model.eval(); va_loss=0.0; preds=[]; labels=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb); loss = criterion(out, yb); va_loss += loss.item()
            preds += out.argmax(1).cpu().tolist(); labels += yb.cpu().tolist()
    print(f"Epoch {ep}/{EPOCHS} | TrainLoss {tr_loss/len(train_loader):.4f} | ValLoss {va_loss/len(val_loader):.4f}")

# 8) Evaluation
print("\n=== Classification Report (Validation) ===")
print(classification_report(labels, preds, target_names=classes))
cm = confusion_matrix(labels, preds)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - MLP (Weighted Loss, 1000 features, 70/30 split)")
plt.show()
