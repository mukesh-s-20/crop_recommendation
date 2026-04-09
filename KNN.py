# ==========================================
# PROFESSIONAL KNN DECISION BOUNDARY (K=5)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# 1️⃣ Load Dataset
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")
df.columns = df.columns.str.strip()

# 2️⃣ Select Only Two Features (Change if needed)
features = ["Temperature", "Humidity"]
X = df[features]

# Encode Crop labels
le = LabelEncoder()
y = le.fit_transform(df["Crop"])

class_names = le.classes_

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Train KNN (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ==========================================
# 🌈 CREATE SMOOTH DECISION BOUNDARY
# ==========================================

# Mesh grid
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 600),
    np.linspace(y_min, y_max, 600)
)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Professional color maps
background_cmap = ListedColormap(plt.cm.Set3.colors[:len(class_names)])
point_cmap = ListedColormap(plt.cm.Dark2.colors[:len(class_names)])

plt.figure(figsize=(10,7))

# Decision surface
plt.contourf(xx, yy, Z, alpha=0.35, cmap=background_cmap)

# Scatter plot for each class separately (for legend)
for i, class_name in enumerate(class_names):
    plt.scatter(
        X_train[y_train == i, 0],
        X_train[y_train == i, 1],
        label=class_name,
        cmap=point_cmap,
        edgecolor='black',
        s=60
    )

plt.title("KNN Decision Boundary (K = 5)", fontsize=16, fontweight='bold')
plt.xlabel(f"{features[0]} (Scaled)", fontsize=12)
plt.ylabel(f"{features[1]} (Scaled)", fontsize=12)
plt.legend(title="Crop Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()

plt.show()
