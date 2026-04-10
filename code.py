# ==========================
# DATA PREPROCESSING & VISUALIZATION
# ==========================
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx", engine="openpyxl")

# Basic info
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# -------------------------
# Check Missing Values
# -------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------
# Correlation Heatmap
# -------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------
# Distribution of Features
# -------------------------
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# -------------------------
# Crop Count Visualization
# -------------------------
plt.figure(figsize=(12,6))
sns.countplot(x="label", data=df)
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.show()
