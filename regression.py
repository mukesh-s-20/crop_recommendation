# ==========================
# REGRESSION MODEL
# ==========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("Crop_Recommendation_Preprocessed.xlsx")

# Predict Rainfall using other ORIGINAL features only
X = df[["Nitrogen", "Phosphorus", "Potassium",
        "Temperature", "Humidity", "pH_Value"]]

y = df["Rainfall"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Visualization
plt.figure(figsize=(6,5))
plt.scatter(y_test, predictions)

# Add ideal prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Regression Model Performance")
plt.show()

