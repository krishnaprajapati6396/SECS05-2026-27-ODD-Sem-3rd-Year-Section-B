import pandas as pd
import numpy as np

# ==========================================
# 1. GENERATE DUMMY SUPERSTORE DATASET
# ==========================================
np.random.seed(42)
n_rows = 100

data = {
    'Order_ID': [f'ORD-{1000 + i}' for i in range(n_rows)],
    'Order_Date': pd.date_range(start='2025-01-01', periods=n_rows, freq='D'),
    'Region': np.random.choice(['East', 'West', 'Central', 'South'], n_rows),
    'Category': np.random.choice(['Furniture', 'Office Supplies', 'Technology'], n_rows),
    'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], n_rows),
    'Sales': np.random.uniform(20.0, 1500.0, n_rows).round(2),
    'Profit': np.random.uniform(-100.0, 500.0, n_rows).round(2),
    'Quantity': np.random.randint(1, 10, n_rows)
}

df = pd.DataFrame(data)

# ==========================================
# 2. CALCULATE DASHBOARD KPIs
# ==========================================
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
total_orders = df['Order_ID'].nunique()
avg_order_value = total_sales / total_orders
overall_profit_margin = (total_profit / total_sales) * 100

print("=" * 50)
print("             KEY PERFORMANCE INDICATORS          ")
print("=" * 50)
print(f"Total Revenue      : ${total_sales:,.2f}")
print(f"Total Profit       : ${total_profit:,.2f}")
print(f"Total Orders       : {total_orders}")
print(f"Average Order Value: ${avg_order_value:,.2f}")
print(f"Profit Margin      : {overall_profit_margin:.2f}%\n")

# ==========================================
# 3. BREAKDOWN SUMMARIES FOR DASHBOARD VISUALS
# ==========================================

# A. Regional Sales Breakdown (For Bar Charts / Maps)
region_summary = df.groupby('Region').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Order_Count=('Order_ID', 'count')
).reset_index()

print("--- Sales & Profit by Region ---")
print(region_summary.to_string(index=False))
print("\n")

# B. Category Breakdown (For Pie / Treemap Charts)
category_summary = df.groupby('Category').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum')
).reset_index()

print("--- Sales by Category ---")
print(category_summary.to_string(index=False))
print("\n")

# ==========================================
# 4. EXPORT PROCESSED DATA FOR BI TOOL IMPORT
# ==========================================
# Exports clean CSV file ready to import into Tableau Public or Power BI Desktop
# Line 70 ko isse replace karein:
df.to_csv('Superstore_Dashboard_Data.csv', index=False)
print("[SUCCESS] Exported 'Superstore_Dashboard_Data.csv' for Tableau/Power BI import.")