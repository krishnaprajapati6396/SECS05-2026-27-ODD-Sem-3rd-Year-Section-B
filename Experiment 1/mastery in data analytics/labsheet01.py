import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set aesthetic styling
sns.set_theme(style="whitegrid")

# ==============================================================================
# Step 1, 2 & 3: Load the Dataset and Inspect
# Using 'HR Employee-Attrition.csv'
# ==============================================================================
filename = "HR Employee-Attrition.csv"
df = pd.read_csv(filename)

print("=" * 60)
print("INITIAL DATA INSPECTION")
print("=" * 60)
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nDescriptive Summary:")
print(df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])

# ==============================================================================
# Step 4: Missing Values Identification & Handling
# ==============================================================================
print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "No missing values found.")

# Industry-standard imputation pipeline for numerical/categorical nulls
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# ==============================================================================
# Step 5: Duplicate Detection & Removal
# ==============================================================================
print("\n" + "=" * 60)
print("DUPLICATE RECORD CHECK")
print("=" * 60)
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows identified: {duplicate_count}")
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed successfully.")

# Drop non-informative constant or ID columns standard in this dataset
cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
print(f"Removed non-informative constant/identifier columns: {cols_to_drop}")

# ==============================================================================
# Step 6: Categorical Encoding
# ==============================================================================
# Binary categorical columns -> Label Encoding
binary_cols = ['Attrition', 'OverTime', 'Gender']
le = LabelEncoder()
for col in binary_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Multi-class nominal columns -> One-Hot Encoding
nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
df = pd.get_dummies(df, columns=[col for col in nominal_cols if col in df.columns], drop_first=True)
print("\nCategorical variables successfully converted via Label and One-Hot Encoding.")

# ==============================================================================
# Step 7: Outlier Detection and IQR Capping (Winsorization)
# ==============================================================================
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['MonthlyIncome'], color='#3498db')
plt.title("Boxplot of Monthly Income (Before IQR Capping)", fontsize=12, weight='bold')
plt.savefig("exp1_boxplot_outliers.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] exp1_boxplot_outliers.png")

# IQR Capping for MonthlyIncome
Q1 = df['MonthlyIncome'].quantile(0.25)
Q3 = df['MonthlyIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_count = ((df['MonthlyIncome'] < lower_bound) | (df['MonthlyIncome'] > upper_bound)).sum()
print(f"MonthlyIncome Outliers detected: {outlier_count}")

# Cap outliers to theoretical boundaries
df['MonthlyIncome'] = np.clip(df['MonthlyIncome'], lower_bound, upper_bound)
print("Outliers capped using IQR threshold boundaries.")

# ==============================================================================
# Step 8: Normalization / Standardization
# ==============================================================================
scaler = StandardScaler()
scale_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
scaled_cols = [f"{col}_Scaled" for col in scale_features]
df[scaled_cols] = scaler.fit_transform(df[scale_features])
print("\nNumerical features standardized using StandardScaler (mean=0, std=1).")

# ==============================================================================
# Step 9: Feature Engineering
# ==============================================================================
# Create discrete Age Group categories
df['AgeGroup'] = pd.cut(
    df['Age'], 
    bins=[17, 30, 45, 65], 
    labels=['Young', 'Middle-Aged', 'Senior']
)

# Create Tenure Stability Ratio (Years at company relative to total working years)
df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

# Categorize Income tiers
df['IncomeCategory'] = pd.qcut(
    df['MonthlyIncome'], 
    q=3, 
    labels=['Low', 'Medium', 'High']
)
print("Engineered new attributes: AgeGroup, TenureRatio, IncomeCategory.")

# ==============================================================================
# Step 10: Export Preprocessed Dataset
# ==============================================================================
output_filename = "Cleaned_HR_Employee_Attrition.csv"
df.to_csv(output_filename, index=False)
print(f"\nPreprocessed dataset exported successfully to '{output_filename}'.")
print(f"Final Cleaned Dataset Dimensions: {df.shape}")