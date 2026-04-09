# ==========================================
# RANDOM FOREST COMPLETE MODEL + VISUALS
# ==========================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# 2️⃣ Load Dataset
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")

# 3️⃣ Clean Column Names
df.columns = df.columns.str.strip()

print("Columns in Dataset:")
print(df.columns)

# 4️⃣ Select Target Column
target_column = "Crop"   # Correct target column

# 5️⃣ Define Features and Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7️⃣ Create Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 8️⃣ Train Model
rf_model.fit(X_train, y_train)

# 9️⃣ Predictions
y_pred = rf_model.predict(X_test)

# 🔟 Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("🌲 RANDOM FOREST RESULTS")
print("==============================")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 1️⃣1️⃣ Confusion Matrix
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,
            annot=False,
            cmap="Greens",
            xticklabels=rf_model.classes_,
            yticklabels=rf_model.classes_)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 1️⃣2️⃣ Feature Importance
importances = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

print("\nTop Important Features:")
print(importance_df)

# ==========================================
# 🌳 VISUALIZE FIRST 3 TREES (Like Image)
# ==========================================

plt.figure(figsize=(25,12))

for i in range(3):
    plt.subplot(1, 3, i+1)
    plot_tree(rf_model.estimators_[i],
              feature_names=X.columns,
              class_names=rf_model.classes_,
              filled=True,
              rounded=True,
              max_depth=2)   # Limit depth for clarity
    plt.title(f"Tree {i+1}")

plt.tight_layout()
plt.show()

# ==========================================
# 🗳 SHOW MAJORITY VOTING (Like Diagram)
# ==========================================

print("\n==============================")
print("🗳 MAJORITY VOTING DEMONSTRATION")
print("==============================")

sample = X_test.iloc[[0]]   # take one sample

for i in range(3):
    tree_pred = rf_model.estimators_[i].predict(sample)
    print(f"Tree {i+1} Prediction:", tree_pred[0])

final_prediction = rf_model.predict(sample)
print("\nFinal Random Forest Prediction:", final_prediction[0])
