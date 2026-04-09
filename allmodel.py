# ==============================
# MODEL COMPARISON - REGRESSION
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")

# Use only original features (avoid data leakage)
X = df[["Nitrogen", "Phosphorus", "Potassium",
        "Temperature", "Humidity", "pH_Value"]]

y = df["Rainfall"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data (important for Linear & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Initialize Models
# ==============================

models = {
    "Linear Regression": LinearRegression(),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# ==============================
# Train and Evaluate
# ==============================

for name, model in models.items():
    
    # Use scaled data only for Linear & KNN
    if name in ["Linear Regression", "KNN Regressor"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    results[name] = {"MSE": mse, "R2": r2}
    
    print(f"\n{name}")
    print("MSE:", mse)
    print("R2 Score:", r2)

# ==============================
# Convert results to DataFrame
# ==============================

results_df = pd.DataFrame(results).T
print("\nModel Comparison Table:")
print(results_df)

# ==============================
# Visualization - R2 Comparison
# ==============================

plt.figure(figsize=(8,5))
plt.bar(results_df.index, results_df["R2"])
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=45)
plt.show()
