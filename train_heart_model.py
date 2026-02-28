import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("datasets/heart.csv")

print("First 5 rows:")
print(df.head())

# Convert target to binary
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# Convert categorical columns
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["fbs"] = df["fbs"].map({"TRUE": 1, "FALSE": 0})
df["exang"] = df["exang"].map({"TRUE": 1, "FALSE": 0})

# Drop unnecessary columns
df = df.drop(columns=["id", "dataset"])

# One-hot encode remaining categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nHeart Disease Model Accuracy:", accuracy * 100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("models/heart_model.pkl", "wb"))
pickle.dump(scaler, open("models/heart_scaler.pkl", "wb"))
pickle.dump(X.columns, open("models/heart_columns.pkl", "wb"))

print("\nHeart model saved successfully!")