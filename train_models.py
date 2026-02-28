import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# 1. Load Dataset
# =====================================

data = pd.read_csv("datasets/diabetes.csv")

print("First 5 rows of dataset:\n")
print(data.head())

# =====================================
# 2. Split Features and Target
# =====================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# =====================================
# 3. Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 4. Feature Scaling (IMPORTANT for SVM)
# =====================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# 5. Train SVM Model
# =====================================

model = SVC(kernel="rbf", C=10, gamma=0.01)
model.fit(X_train, y_train)

# =====================================
# 6. Evaluate Model
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nSVM Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# 7. Save Model + Scaler
# =====================================

pickle.dump(model, open("models/diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("models/diabetes_scaler.pkl", "wb"))

print("\nModel and Scaler saved successfully!")