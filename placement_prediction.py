import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "dataset/student_placement_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "salary_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

# Clean
df = df.dropna()

# Speed-up for development
if len(df) > 20000:
    df = df.sample(20000, random_state=42)

# Target
target_col = "salary_package_lpa"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# Drop leakage column and target
drop_cols = [target_col]
if "placement_status" in df.columns:
    drop_cols.append("placement_status")

X = df.drop(columns=drop_cols)
y = df[target_col]

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# Plot actual vs predicted
plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Salary (LPA)")
plt.ylabel("Predicted Salary (LPA)")
plt.title("Actual vs Predicted Salary")
plt.show()

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(X.columns.tolist(), FEATURES_PATH)

print("\nModel saved successfully in models/ folder.")