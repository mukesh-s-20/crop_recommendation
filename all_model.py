# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")
data.columns = data.columns.str.strip()

X = data.drop("Crop", axis=1)
y = data["Crop"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# INITIALIZE MODELS 
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='rbf')
}

accuracy_results = {}

# ==========================================
# TRAIN & EVALUATE
# ==========================================

for name, model in models.items():

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy_results[name] = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy_results[name]:.4f}")

# ==========================================
# SORT RESULTS
# ==========================================

results_df = pd.DataFrame({
    "Model": accuracy_results.keys(),
    "Accuracy": accuracy_results.values()
}).sort_values(by="Accuracy", ascending=False)

print("\nModel Comparison:")
print(results_df)

# ==========================================
# PLOT GRAPH (ALL SAME COLOR)
# ==========================================

plt.figure(figsize=(9,5))
bars = plt.bar(results_df["Model"], results_df["Accuracy"])

# Display accuracy values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f"{height:.3f}",
             ha='center',
             va='bottom')

plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Comparison of Classification Models")
plt.tight_layout()
plt.show()

print("\nBest Model:", results_df.iloc[0]["Model"])