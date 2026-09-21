import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
# -------------------------------------------------------------
# 0. Setup Output Directory
# -------------------------------------------------------------
OUTPUT_DIR = "experiment_10_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def save_and_show(fig, filename):
    """Utility helper to save an image distinctly before displaying it."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[Saved]: {filepath}")
    plt.show()
    plt.close(fig)

# -------------------------------------------------------------
# 1. Synthetic Representative HR Data Generation
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 1200

ages = np.random.randint(22, 60, n_samples)
overtime = np.random.choice([0, 1], size=n_samples, p=[0.72, 0.28])
monthly_income = np.random.normal(6500, 2500, n_samples).clip(1500, 20000)
years_at_company = np.random.randint(0, 20, n_samples)
job_satisfaction = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.15, 0.20, 0.35, 0.30])

logits = (
    -1.8 
    - 0.03 * (ages - 35) 
    + 1.4 * overtime 
    - 0.00035 * (monthly_income - 6500) 
    - 0.5 * (job_satisfaction - 2.5)
)
attrition_prob = 1 / (1 + np.exp(-logits))
attrition = (np.random.rand(n_samples) < attrition_prob).astype(int)

df = pd.DataFrame({
    'Age': ages,
    'OverTime': overtime,
    'MonthlyIncome': monthly_income,
    'YearsAtCompany': years_at_company,
    'JobSatisfaction': job_satisfaction,
    'Attrition': attrition
})

# -------------------------------------------------------------
# 2. Statistical Analysis & Hypothesis Testing
# -------------------------------------------------------------
leavers_income = df[df['Attrition'] == 1]['MonthlyIncome']
stayers_income = df[df['Attrition'] == 0]['MonthlyIncome']

t_stat, p_val = stats.ttest_ind(leavers_income, stayers_income, equal_var=False)
print("=== STATISTICAL HYPOTHESIS TEST (Two-Sample t-Test) ===")
print(f"Leavers Mean Income: ${leavers_income.mean():.2f}")
print(f"Stayers Mean Income: ${stayers_income.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4e}\n")

# -------------------------------------------------------------
# 3. Model Training & Evaluation
# -------------------------------------------------------------
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

print("=== MODEL PERFORMANCE METRICS ===")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Attrited']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}\n")

# -------------------------------------------------------------
# 4. Individual Plot Generation & Automatic Saving
# -------------------------------------------------------------

# Image 1: Income Disparity Boxplot
fig1, ax1 = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', ax=ax1, palette=['#4C72B0', '#C44E52'])
ax1.set_xticklabels(['Retained', 'Attrited'])
ax1.set_title('1. Income Gap by Attrition Status', fontweight='bold', fontsize=12)
ax1.set_ylabel('Monthly Income ($)')
save_and_show(fig1, "01_income_gap_boxplot.png")

# Image 2: OverTime Attrition Rate
fig2, ax2 = plt.subplots(figsize=(7, 5))
ot_attr = df.groupby('OverTime')['Attrition'].mean().reset_index()
sns.barplot(data=ot_attr, x='OverTime', y='Attrition', ax=ax2, palette=['#55A868', '#C44E52'])
ax2.set_xticklabels(['No Overtime', 'Frequent Overtime'])
ax2.set_ylabel('Attrition Rate (%)')
ax2.set_title('2. Burnout Factor: OverTime vs Turnover Rate', fontweight='bold', fontsize=12)
save_and_show(fig2, "02_overtime_turnover_bar.png")

# Image 3: Correlation Matrix Heatmap
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax3)
ax3.set_title('3. Feature Correlation Matrix', fontweight='bold', fontsize=12)
save_and_show(fig3, "03_correlation_matrix.png")

# Image 4: Model Feature Importances
fig4, ax4 = plt.subplots(figsize=(8, 5))
feat_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
feat_importances.plot(kind='barh', ax=ax4, color='#4C72B0')
ax4.set_title('4. Key Drivers: Random Forest Feature Importance', fontweight='bold', fontsize=12)
ax4.set_xlabel('Relative Importance Weight')
save_and_show(fig4, "04_feature_importance.png")

# Image 5: Confusion Matrix
fig5, ax5 = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Retained', 'Attrited'],
            yticklabels=['Retained', 'Attrited'])
ax5.set_title('5. Random Forest Confusion Matrix', fontweight='bold', fontsize=12)
ax5.set_xlabel('Predicted Label')
ax5.set_ylabel('True Label')
save_and_show(fig5, "05_confusion_matrix.png")

# Image 6: ROC Curve
fig6, ax6 = plt.subplots(figsize=(7, 5))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax6.plot(fpr, tpr, color='#4C72B0', lw=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
ax6.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
ax6.set_xlim([0.0, 1.0])
ax6.set_ylim([0.0, 1.05])
ax6.set_xlabel('False Positive Rate')
ax6.set_ylabel('True Positive Rate')
ax6.set_title('6. Receiver Operating Characteristic (ROC)', fontweight='bold', fontsize=12)
ax6.legend(loc="lower right")
save_and_show(fig6, "06_roc_curve.png")

print(f"\nAll 6 visualization files have been generated and saved in './{OUTPUT_DIR}/'.")