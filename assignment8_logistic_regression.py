# Assignment 8: Logistic Regression on Breast Cancer Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# ---------- Step 2: Load Data ----------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)
print("\nFirst five records:\n", X.head())
print("\nTarget classes:", data.target_names)
print("0 = Malignant, 1 = Benign")

# ---------- Step 3: Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ---------- Step 4: Standardization ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------- Step 5: Train Logistic Regression ----------
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)
print("\nLogistic Regression model trained successfully!")

# ---------- Step 6: Predictions ----------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability for class 1

# ---------- Step 7: Accuracy ----------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ---------- Step 8: Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()

# ---------- Step 9: Classification Report ----------
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ---------- Step 10: ROC Curve ----------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print("ROC-AUC Score:", round(auc_score, 4))
