import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


print("Running Breast Cancer Training Script...\n")

# ==============================
# Load Dataset
# ==============================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)
print("\nClass Distribution:")
print(y.value_counts())


# ==============================
# Initial Train (All Features)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler_full = StandardScaler()
X_train_scaled = scaler_full.fit_transform(X_train)
X_test_scaled = scaler_full.transform(X_test)

model_full = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model_full.fit(X_train_scaled, y_train)

# ==============================
# Feature Importance
# ==============================

feature_importances = pd.Series(
    model_full.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 12 Important Features:")
print(feature_importances.head(12))

# Select top 12
top_features = feature_importances.head(12).index.tolist()

print("\nSelected Features for Final Model:")
print(top_features)


# ==============================
# Retrain Using Top 12 Features
# ==============================

X_top = X[top_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_top,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Breast Cancer Model Accuracy (Top 12 Features):", accuracy * 100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ==============================
# Save Model
# ==============================

pickle.dump(model, open("models/breast_cancer_model.pkl", "wb"))
pickle.dump(scaler, open("models/breast_cancer_scaler.pkl", "wb"))
pickle.dump(top_features, open("models/breast_cancer_columns.pkl", "wb"))

print("\nBreast Cancer model (Top 12 features) saved successfully!")