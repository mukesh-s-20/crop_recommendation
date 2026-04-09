# ==========================================
# CROP RECOMMENDATION - COMPLETE CLASSIFIER
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx", engine="openpyxl")

print("Dataset Loaded Successfully!\n")
print("Columns in dataset:\n", df.columns)

# ==========================================
# 2. AUTO-DETECT TARGET COLUMN
# ==========================================

possible_targets = ["label", "Label", "crop", "Crop"]

target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise Exception("Target column not found! Please check your dataset column name.")

print("\nTarget Column Detected:", target_column)

# ==========================================
# 3. REMOVE NORMALIZED DUPLICATES (Optional Safety)
# ==========================================

# If normalized columns exist, drop them
cols_to_drop = [col for col in df.columns if "normalized" in col.lower()]
df = df.drop(columns=cols_to_drop)

# ==========================================
# 4. SPLIT FEATURES & TARGET
# ==========================================

X = df.drop(target_column, axis=1)
y = df[target_column]

# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 6. FEATURE SCALING
# ==========================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 7. KNN - FIND BEST K
# ==========================================

k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_temp))

# Plot K vs Accuracy
plt.figure()
plt.plot(k_values, accuracies)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K Value")
plt.show()

best_k = k_values[np.argmax(accuracies)]
print("Best K Value:", best_k)

# Final KNN Model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# ==========================================
# 8. LOGISTIC REGRESSION
# ==========================================

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ==========================================
# 9. SVM
# ==========================================

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# ==========================================
# 10. RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ==========================================
# 11. ACCURACY COMPARISON
# ==========================================

results = {
    "KNN": accuracy_score(y_test, y_pred_knn),
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "SVM": accuracy_score(y_test, y_pred_svm),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
}

print("\nModel Accuracies:")
for model, acc in results.items():
    print(f"{model}: {round(acc * 100, 2)} %")

# Save results
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
results_df.to_csv("Model_Results.csv", index=False)
print("\nResults saved as Model_Results.csv")

# Plot Model Comparison
plt.figure()
plt.bar(results.keys(), results.values())
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=30)
plt.show()

# ==========================================
# 12. CONFUSION MATRICES
# ==========================================

models_predictions = {
    "KNN": y_pred_knn,
    "Logistic Regression": y_pred_lr,
    "SVM": y_pred_svm,
    "Random Forest": y_pred_rf,
}

for name, predictions in models_predictions.items():
    plt.figure()
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==========================================
# 13. CLASSIFICATION REPORTS
# ==========================================

print("\nClassification Reports:\n")

for name, predictions in models_predictions.items():
    print("===== ", name, " =====")
    print(classification_report(y_test, predictions))