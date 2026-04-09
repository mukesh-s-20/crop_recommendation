# ==========================================
# DECISION TREE - COMPLETE MODEL
# ==========================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2️⃣ Load Dataset
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")

# 3️⃣ Clean Column Names
df.columns = df.columns.str.strip()

print("Columns in Dataset:")
print(df.columns)

# 4️⃣ Set Target Column
target_column = "Crop"   # Crop is the output class

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

# 7️⃣ Create Decision Tree Model
dt_model = DecisionTreeClassifier(
    criterion="gini",   # you can also use "entropy"
    max_depth=None,
    random_state=42
)

# 8️⃣ Train Model
dt_model.fit(X_train, y_train)

# 9️⃣ Predictions
y_pred = dt_model.predict(X_test)

# 🔟 Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("🌳 DECISION TREE RESULTS")
print("==============================")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 1️⃣1️⃣ Confusion Matrix
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,
            annot=False,
            cmap="Blues",
            xticklabels=dt_model.classes_,
            yticklabels=dt_model.classes_)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 1️⃣2️⃣ Feature Importance
importances = dt_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance - Decision Tree")
plt.tight_layout()
plt.show()

print("\nTop Important Features:")
print(importance_df)

# ==========================================
# 🌳 TREE VISUALIZATION
# ==========================================

plt.figure(figsize=(20,10))
plot_tree(dt_model,
          feature_names=X.columns,
          class_names=dt_model.classes_,
          filled=True,
          rounded=True,
          max_depth=3)  # limit depth for clarity

plt.title("Decision Tree Visualization")
plt.show()
