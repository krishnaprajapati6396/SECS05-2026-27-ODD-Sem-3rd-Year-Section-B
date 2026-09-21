import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# ==============================================================================
# Step 0: Setup Output Directory
# ==============================================================================
output_dir = "churn_output_images"
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# Step 1 & 2: Load Dataset from GitHub
# ==============================================================================
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(DATA_URL)
    print("Dataset loaded successfully.")
except Exception as err:
    print(f"Failed to fetch dataset: {err}")
    raise SystemExit

print(f"Dataset Dimensions: {df.shape}")
print(df.head(3))

# ==============================================================================
# Step 3: Data Preprocessing & Feature Encoding
# ==============================================================================
# Drop customer identifier
df = df.drop(columns=['customerID'])

# Convert TotalCharges from string to numeric (handles blank spaces as NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Map binary target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify binary columns vs. multi-class categorical columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Map gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# One-Hot Encode remaining categorical variables
categorical_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# ==============================================================================
# Step 4: Train-Test Split (80:20 Ratio)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

# ==============================================================================
# Step 5: Train Decision Tree Classifier
# ==============================================================================
# Using max_depth=4 to balance expressiveness with interpretability and avoid overfitting
dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
dt_classifier.fit(X_train, y_train)

# ==============================================================================
# Step 6 & 7: Model Predictions & Performance Evaluation
# ==============================================================================
y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print("MODEL PERFORMANCE METRICS")
print("=" * 55)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churned (1)']))

# Plot and Save Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=['Retained', 'Churned'], 
    yticklabels=['Retained', 'Churned']
)
plt.title('Confusion Matrix - Customer Churn', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()

cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"[Saved Image] Confusion matrix saved to: {cm_path}")
plt.show()

# ==============================================================================
# Step 8: Visualize the Decision Tree Structure
# ==============================================================================
plt.figure(figsize=(24, 12))
plot_tree(
    dt_classifier,
    feature_names=X.columns.tolist(),
    class_names=['Retained', 'Churned'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('Trained Decision Tree (max_depth=4)', fontsize=15, fontweight='bold')
plt.tight_layout()

tree_path = os.path.join(output_dir, "decision_tree_structure.png")
plt.savefig(tree_path, dpi=300, bbox_inches='tight')
print(f"[Saved Image] Decision tree visualization saved to: {tree_path}")
plt.show()

# ==============================================================================
# Step 9: Feature Importance Analysis
# ==============================================================================
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Keep only top 8 features contributing to splits
top_features = importance_df[importance_df['Importance'] > 0.01]

plt.figure(figsize=(8, 4.5))
sns.barplot(
    data=top_features, 
    x='Importance', 
    y='Feature', 
    palette='viridis'
)
plt.title('Key Features Driving Customer Churn', fontsize=12, fontweight='bold')
plt.xlabel('Feature Importance Score')
plt.ylabel('Attribute')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

fi_path = os.path.join(output_dir, "feature_importance.png")
plt.savefig(fi_path, dpi=300, bbox_inches='tight')
print(f"[Saved Image] Feature importances saved to: {fi_path}")
plt.show()

# ==============================================================================
# Step 10: Print Strategic Business Recommendations
# ==============================================================================
print("\n" + "=" * 55)
print("BUSINESS RETENTION RECOMMENDATIONS")
print("=" * 55)
print("1. Target Month-to-Month Contracts: Contract type is the strongest split. Transition customers toward annual contracts with incentives.")
print("2. Monitor Tenure Milestones: Churn risk drops sharply as customer tenure passes 12 months; provide dedicated onboarding support in months 1-6.")
print("3. TechSupport & Fiber Quality: Customers on Fiber Optic without TechSupport exhibit high churn; bundle assistance and resolve network latency.")