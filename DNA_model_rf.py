import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Load merged CNV + clinical stage data
data_path = "BRCA_CNV_stage.csv"  # change if needed
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")

# Step 2: Prepare features (X) and labels (y)
X = df.drop(columns=["patient_id", "stage_clean"])
y = df["stage_clean"]

# Encode stage labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features (important for some models, though RF is less sensitive)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step 4: Train Random Forest model
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,       # number of trees
    max_depth=None,         # let trees expand until pure
    class_weight="balanced",# handle class imbalance
    random_state=42,
    n_jobs=-1               # use all CPUs
)
rf.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = rf.predict(X_test)
print("\n=== Model: RF Classifier ===")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 6: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix_rf.png", dpi=300, bbox_inches="tight")
plt.show()
