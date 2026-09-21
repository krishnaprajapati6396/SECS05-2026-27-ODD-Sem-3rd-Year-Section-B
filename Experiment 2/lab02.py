import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# ==============================================================================
# Step 1 & 2: Load the Dataset
# Using 'HR Employee-Attrition.csv'
# ==============================================================================
filename = "HR Employee-Attrition.csv"
df = pd.read_csv(filename)

# ==============================================================================
# Step 3: Structural & Summary Exploration
# ==============================================================================
print("=" * 60)
print("1. DATASET DIMENSIONS & INFO")
print("=" * 60)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
print(df.info())

print("\n" + "=" * 60)
print("2. NUMERICAL SUMMARY STATISTICS")
print("=" * 60)
print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].head(10))

print("\n" + "=" * 60)
print("3. TARGET CLASS DISTRIBUTION (Attrition)")
print("=" * 60)
print(df['Attrition'].value_counts(normalize=True).round(4) * 100)

# ==============================================================================
# Step 4: Univariate Numerical Distribution (Histogram + KDE)
# ==============================================================================
plt.figure(figsize=(8, 5))
sns.histplot(df['MonthlyIncome'], kde=True, bins=30, color='#1f77b4')
plt.title("Distribution of Monthly Income", fontsize=14, weight='bold')
plt.xlabel("Monthly Income ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.savefig("eda_1_monthly_income_dist.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] eda_1_monthly_income_dist.png")

# ==============================================================================
# Step 5: Categorical Analysis (Count Plot)
# ==============================================================================
plt.figure(figsize=(9, 5))
order = df['JobRole'].value_counts().index
sns.countplot(data=df, y='JobRole', order=order, palette='mako')
plt.title("Employee Distribution Across Job Roles", fontsize=14, weight='bold')
plt.xlabel("Number of Employees", fontsize=12)
plt.ylabel("Job Role", fontsize=12)
plt.savefig("eda_2_job_role_counts.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] eda_2_job_role_counts.png")

# ==============================================================================
# Step 6: Correlation Matrix Heatmap
# ==============================================================================
plt.figure(figsize=(8, 6))
num_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole']
corr_matrix = df[num_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Key Numerical Variables", fontsize=14, weight='bold')
plt.savefig("eda_3_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] eda_3_correlation_matrix.png")

# ==============================================================================
# Step 7: Bivariate Outlier & Group Comparison (Box Plot)
# ==============================================================================
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', palette='Set2')
plt.title("Monthly Income Distribution by Attrition Status", fontsize=14, weight='bold')
plt.xlabel("Attrition", fontsize=12)
plt.ylabel("Monthly Income ($)", fontsize=12)
plt.savefig("eda_4_boxplot_income_attrition.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] eda_4_boxplot_income_attrition.png")

# ==============================================================================
# Step 8: Multivariate Analysis (Scatter Plot with Hue)
# ==============================================================================
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x='TotalWorkingYears',
    y='MonthlyIncome',
    hue='Attrition',
    alpha=0.7,
    palette={'Yes': '#d62728', 'No': '#1f77b4'}
)
plt.title("Income vs. Total Working Years Segmented by Attrition", fontsize=14, weight='bold')
plt.xlabel("Total Working Years", fontsize=12)
plt.ylabel("Monthly Income ($)", fontsize=12)
plt.savefig("eda_5_multivariate_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] eda_5_multivariate_scatter.png")

print("\nEDA processing complete. All 5 visualization figures saved.")