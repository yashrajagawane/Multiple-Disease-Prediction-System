import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Load Dataset
# =========================
df = pd.read_csv("datasets/kidney.csv")

print("First 5 rows:")
print(df.head())

# =========================
# Drop ID column
# =========================
df = df.drop(columns=["id"])

# =========================
# Clean Target
# =========================
df["classification"] = df["classification"].str.strip()
df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

# =========================
# Replace ? with NaN
# =========================
df.replace("?", np.nan, inplace=True)

# =========================
# Strip spaces in all string columns
# =========================
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()

# =========================
# Convert categorical columns
# =========================
binary_map = {
    "yes": 1, "no": 0,
    "normal": 1, "abnormal": 0,
    "present": 1, "notpresent": 0,
    "good": 1, "poor": 0
}

for col in df.columns:
    df[col] = df[col].replace(binary_map)

# =========================
# Convert numeric columns
# =========================
df = df.apply(pd.to_numeric, errors='coerce')

# =========================
# Fill Missing Values
# =========================
df = df.fillna(df.median())

# =========================
# Split Features & Target
# =========================
X = df.drop("classification", axis=1)
y = df["classification"]

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Scale
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Train Model
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Evaluate
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nKidney Disease Model Accuracy:", accuracy * 100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# Save Model
# =========================
pickle.dump(model, open("models/kidney_model.pkl", "wb"))
pickle.dump(scaler, open("models/kidney_scaler.pkl", "wb"))
pickle.dump(X.columns, open("models/kidney_columns.pkl", "wb"))

print("\nKidney model saved successfully!")