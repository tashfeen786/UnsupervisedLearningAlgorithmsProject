import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("🌸 Iris Classification using Logistic Regression (Binary)")

# Step 1: Load Iris Data
iris = load_iris()
X = iris.data
y = iris.target
y_binary = (y == 0).astype(int)  # Convert to binary: Class 0 vs Others

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Step 3: Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Step 6: Evaluation
st.subheader("📊 Classification Metrics")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")
st.write(f"**ROC AUC Score:** {roc_auc:.2f}")

# Step 7: Confusion Matrix
st.subheader("🧾 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig1)

# Step 8: ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_val)
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
ax2.legend(loc="lower right")
st.pyplot(fig2)

# Step 9: Full Classification Report
st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))
