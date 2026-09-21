import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# ==============================================================================
# Step 1 & 2: Load the Dataset and Import Libraries
# ==============================================================================
filename = "HR Employee-Attrition.csv"
df = pd.read_csv(filename)

print(f"Dataset successfully loaded. Shape: {df.shape}\n")

# ==============================================================================
# Step 3: Descriptive Statistics for Numerical Features
# ==============================================================================
numerical_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']

# Mean, Median, Variance, Standard Deviation
desc_stats = df[numerical_cols].agg(['mean', 'median', 'var', 'std']).T

# Compute Mode
desc_stats['mode'] = [df[col].mode()[0] for col in numerical_cols]

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(desc_stats.round(2))
print()

# ==============================================================================
# Step 4: Pearson Correlation Analysis & Automatic Plot Saving
# ==============================================================================
corr_matrix = df[numerical_cols].corr(method='pearson')

print("=" * 60)
print("PEARSON CORRELATION MATRIX")
print("=" * 60)
print(corr_matrix.round(3))
print()

# Generate and auto-save Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Employee Metrics")
plt.tight_layout()

# Automatically save before showing
plt.savefig("correlation_matrix.png", dpi=300, bbox_inches='tight')
print("[INFO] Figure saved automatically as 'correlation_matrix.png'")

# Non-blocking show prevents terminal execution from halting
plt.show(block=False)

# ==============================================================================
# Step 5 & 6: Independent Two-Sample t-Test
# Business Question: Does monthly income differ between employees who stay vs leave?
# H0: Mean MonthlyIncome(Attrition == 'Yes') == Mean MonthlyIncome(Attrition == 'No')
# H1: Mean MonthlyIncome(Attrition == 'Yes') != Mean MonthlyIncome(Attrition == 'No')
# ==============================================================================
income_left = df[df['Attrition'] == 'Yes']['MonthlyIncome']
income_stayed = df[df['Attrition'] == 'No']['MonthlyIncome']

# Welch's t-test (equal_var=False)
t_stat, p_val_ttest = stats.ttest_ind(income_left, income_stayed, equal_var=False)

print("\n" + "=" * 60)
print("HYPOTHESIS TEST 1: TWO-SAMPLE T-TEST")
print("=" * 60)
print(f"Mean Income (Left):   ${income_left.mean():.2f}")
print(f"Mean Income (Stayed): ${income_stayed.mean():.2f}")
print(f"t-statistic:          {t_stat:.4f}")
print(f"p-value:              {p_val_ttest:.4e}")

if p_val_ttest < 0.05:
    print("Decision: Reject the Null Hypothesis (H0).")
    print("Conclusion: Significant difference in monthly income between employees who leave and stay.")
else:
    print("Decision: Fail to reject the Null Hypothesis (H0).")
print()

# ==============================================================================
# Step 7: One-Way ANOVA
# Business Question: Does monthly income vary across different Job Roles?
# H0: Mean MonthlyIncome is equal across all Job Roles
# H1: At least one Job Role has a different mean MonthlyIncome
# ==============================================================================
role_groups = [group['MonthlyIncome'].values for _, group in df.groupby('JobRole')]
f_stat, p_val_anova = stats.f_oneway(*role_groups)

print("=" * 60)
print("HYPOTHESIS TEST 2: ONE-WAY ANOVA")
print("=" * 60)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value:     {p_val_anova:.4e}")

if p_val_anova < 0.05:
    print("Decision: Reject the Null Hypothesis (H0).")
    print("Conclusion: Monthly income differs significantly across job roles.")
else:
    print("Decision: Fail to reject the Null Hypothesis (H0).")
print()

# ==============================================================================
# Step 8 & 9: Simple Linear Regression (OLS)
# Dependent (y): MonthlyIncome, Independent (X): TotalWorkingYears
# ==============================================================================
X = sm.add_constant(df['TotalWorkingYears'])
y = df['MonthlyIncome']

model = sm.OLS(y, X).fit()

print("=" * 60)
print("LINEAR REGRESSION ANALYSIS")
print("=" * 60)
print(model.summary())

# Plot and auto-save Regression Line
plt.figure(figsize=(8, 5))
sns.regplot(
    x='TotalWorkingYears', 
    y='MonthlyIncome', 
    data=df, 
    scatter_kws={'alpha': 0.3}, 
    line_kws={'color': 'red'}
)
plt.title("Regression: Monthly Income vs Total Working Years")
plt.xlabel("Total Working Years")
plt.ylabel("Monthly Income ($)")
plt.tight_layout()

plt.savefig("regression_line.png", dpi=300, bbox_inches='tight')
print("[INFO] Figure saved automatically as 'regression_line.png'")

# Display all open figures
plt.show()