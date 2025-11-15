import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_excel("Employees.xlsx")

# Select valid features
features = [
    "Years", "Department", "Country", "Center",
    "Job Rate", "Sick Leaves", "Unpaid Leaves", "Overtime Hours"
]
target = "Annual Salary"

# Keep only existing columns
df = df[features + [target]]

# Convert categorical features
df = pd.get_dummies(df, columns=["Department", "Country", "Center"], drop_first=True)

# Split data
X = df.drop(target, axis=1)
y = df[target]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model + scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("Model Training Completed.")

