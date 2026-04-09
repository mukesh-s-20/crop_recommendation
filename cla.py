# =====================================
# IMPORT LIBRARIES
# =====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# =====================================
# LOAD DATASET
# =====================================

data = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")

data.columns = data.columns.str.strip()

print("Columns:", data.columns)

# TARGET COLUMN IS "Crop"
X = data.drop("Crop", axis=1)
y = data["Crop"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# FEATURE SCALING
# =====================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# INITIALIZE MODELS
# =====================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

accuracies = {}

# =====================================
# TRAIN & TEST MODELS
# =====================================

for name, model in models.items():

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# =====================================
# ACCURACY COMPARISON GRAPH
# =====================================

plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# =====================================
# KNN K VS ACCURACY GRAPH
# =====================================

k_range = range(1, 11)
knn_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    knn_scores.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(k_range, knn_scores)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN: K vs Accuracy")
plt.xticks(k_range)
plt.show()

# =====================================
# RANDOM FOREST FEATURE IMPORTANCE
# =====================================

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

plt.figure()
plt.barh(X.columns, rf.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

# =====================================
# BEST MODEL
# =====================================

best_model_name = max(accuracies, key=accuracies.get)
print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

if best_model_name in ["Logistic Regression", "KNN", "SVM"]:
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))