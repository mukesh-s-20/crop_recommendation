# ============================================================
# COMPLETE CLUSTERING (KMEANS + HIERARCHICAL + DBSCAN)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------------

file_path = r"Crop_Recommendation_Preprocessed.xlsx"
df = pd.read_excel(file_path)

print("Columns in dataset:")
print(df.columns)

# ------------------------------------------------------------
# 2️⃣ Separate Features and Crop Name
# ------------------------------------------------------------

X = df.drop("Crop", axis=1)
crop_names = df["Crop"]

# Keep only numerical columns
X = X.select_dtypes(include=[np.number])

# ------------------------------------------------------------
# 3️⃣ Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 4️⃣ PCA for 2D Visualization
# ------------------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ============================================================
# 🔵 1. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10,7))
for cluster in np.unique(kmeans_labels):
    indices = kmeans_labels == cluster
    plt.scatter(
        X_pca[indices, 0],
        X_pca[indices, 1],
        label=f"KMeans Cluster {cluster}"
    )

plt.title("K-Means Clustering (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 🟢 2. HIERARCHICAL CLUSTERING
# ============================================================

hierarchical = AgglomerativeClustering(n_clusters=5)
hier_labels = hierarchical.fit_predict(X_scaled)

plt.figure(figsize=(10,7))
for cluster in np.unique(hier_labels):
    indices = hier_labels == cluster
    plt.scatter(
        X_pca[indices, 0],
        X_pca[indices, 1],
        label=f"Hierarchical Cluster {cluster}"
    )

plt.title("Hierarchical Clustering (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 🔴 3. DBSCAN CLUSTERING
# ============================================================

dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(10,7))
for cluster in np.unique(dbscan_labels):
    indices = dbscan_labels == cluster
    plt.scatter(
        X_pca[indices, 0],
        X_pca[indices, 1],
        label=f"DBSCAN Cluster {cluster}"
    )

plt.title("DBSCAN Clustering (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()