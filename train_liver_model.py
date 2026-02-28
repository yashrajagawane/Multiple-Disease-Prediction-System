import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# =========================
# Load Dataset
# =========================
df = pd.read_csv("datasets/liver.csv")

# Convert target
df["is_patient"] = df["is_patient"].apply(lambda x: 1 if x == 1 else 0)

# Encode gender
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# Handle missing values
df = df.fillna(df.median())

# =========================
# Handle Class Imbalance (Oversampling)
# =========================
majority = df[df.is_patient == 1]
minority = df[df.is_patient == 0]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

df = pd.concat([majority, minority_upsampled])

# =========================
# Split Features & Target
# =========================
X = df.drop("is_patient", axis=1)
y = df["is_patient"]

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
# Train Better Model
# =========================
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Evaluate
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nImproved Liver Model Accuracy:", accuracy * 100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# Save
# =========================
pickle.dump(model, open("models/liver_model.pkl", "wb"))
pickle.dump(scaler, open("models/liver_scaler.pkl", "wb"))
pickle.dump(X.columns, open("models/liver_columns.pkl", "wb"))

print("\nImproved liver model saved successfully!")