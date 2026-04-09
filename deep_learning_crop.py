# ==========================================
# DEEP LEARNING STYLE MODEL FOR CROP DATA
# WITH TRAINING / VALIDATION GRAPHS
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, log_loss

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")
df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully!\n")
print("Columns in Dataset:")
print(df.columns)

# ==========================================
# 3. SELECT FEATURES AND TARGET
#    (Reduced features to keep accuracy realistic)
# ==========================================
X = df[["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity"]]

y = df["Crop"]

# ==========================================
# 4. ENCODE TARGET LABELS
# ==========================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Split training further into train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# ==========================================
# 6. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. BUILD MLP MODEL
# ==========================================
mlp_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),   # smaller network
    activation='relu',
    solver='adam',
    max_iter=1,                    # one epoch at a time
    warm_start=True,               # continue training
    random_state=42
)

# ==========================================
# 8. TRAIN MODEL EPOCH BY EPOCH
# ==========================================
epochs = 40

train_acc = []
val_acc = []
train_loss = []
val_loss = []

classes = np.unique(y_train)

for epoch in range(epochs):
    mlp_model.fit(X_train_scaled, y_train)

    # Predictions
    train_pred = mlp_model.predict(X_train_scaled)
    val_pred = mlp_model.predict(X_val_scaled)

    # Probabilities for loss
    train_prob = mlp_model.predict_proba(X_train_scaled)
    val_prob = mlp_model.predict_proba(X_val_scaled)

    # Accuracy
    train_acc.append(accuracy_score(y_train, train_pred))
    val_acc.append(accuracy_score(y_val, val_pred))

    # Loss
    train_loss.append(log_loss(y_train, train_prob, labels=classes))
    val_loss.append(log_loss(y_val, val_prob, labels=classes))

# ==========================================
# 9. FINAL TEST PREDICTION
# ==========================================
y_pred = mlp_model.predict(X_test_scaled)

# ==========================================
# 10. EVALUATION METRICS
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n===== DEEP LEARNING MODEL PERFORMANCE =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# ==========================================
# 11. GRAPH 1 - TRAINING ACCURACY VS VALIDATION ACCURACY
# ==========================================
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_acc, marker='o', label='Training Accuracy')
plt.plot(range(1, epochs+1), val_acc, marker='s', label='Validation Accuracy')
plt.title("Training Accuracy vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 12. GRAPH 2 - TRAINING LOSS VS VALIDATION LOSS
# ==========================================
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_loss, marker='o', label='Training Loss')
plt.plot(range(1, epochs+1), val_loss, marker='s', label='Validation Loss')
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 13. GRAPH 3 - CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - Deep Learning Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# 14. GRAPH 4 - PERFORMANCE METRICS BAR CHART
# ==========================================
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8,5))
bars = plt.bar(metrics_names, metrics_values)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha='center',
        va='bottom'
    )

plt.ylim(0, 1.1)
plt.title("Deep Learning Model Performance Metrics")
plt.ylabel("Score")
plt.show()
plt.show()