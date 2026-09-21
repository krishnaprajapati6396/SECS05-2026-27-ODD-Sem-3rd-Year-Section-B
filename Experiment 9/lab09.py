import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"

# 1. Load Data (Assumes global_superstore.csv is present)
# df = pd.read_csv('global_superstore.csv', encoding='latin-1')

# --- Mock Data Representation for Reproducibility ---
data = {
    'Market': ['APAC', 'EU', 'US', 'LATAM', 'EMEA', 'Africa'],
    'Sales': [3585743, 2938330, 2297200, 2164605, 806161, 783773],
    'Profit': [436000, 372800, 286397, 221644, 43897, 88860],
    'Avg_Discount': [0.15, 0.10, 0.16, 0.14, 0.44, 0.16]
}
df_market = pd.DataFrame(data)
df_market['Profit_Margin'] = (df_market['Profit'] / df_market['Sales']) * 100

subcat_data = {
    'Sub_Category': ['Tables', 'Bookcases', 'Supplies', 'Fasteners', 'Copiers', 'Phones'],
    'Profit': [-64083, -3472, -1189, 11500, 258000, 216717],
    'Sales': [757000, 507000, 243000, 83000, 1509000, 1706000]
}
df_subcat = pd.DataFrame(subcat_data)

# --- Visual Storytelling Dashboard (2x2 Multi-Panel) ---
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Panel 1: Context (Revenue vs Profit by Market)
sns.barplot(data=df_market, x='Market', y='Sales', color='#4C72B0', ax=axes[0, 0], label='Revenue')
sns.barplot(data=df_market, x='Market', y='Profit', color='#55A868', ax=axes[0, 0], label='Net Profit')
axes[0, 0].set_title('1. Market Scale: APAC and EU Drive Majority Volume', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('USD ($)')
axes[0, 0].legend()

# Panel 2: Complication (EMEA Margin Collapse vs Discounting)
sns.barplot(data=df_market, x='Market', y='Profit_Margin', palette='Blues_r', ax=axes[0, 1])
axes[0, 1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0, 1].set_title('2. Margin Vulnerability: EMEA Profit Margin Lags at 5.4%', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Profit Margin (%)')

# Panel 3: Root Cause (Sub-Category Bleed)
colors = ['#C44E52' if x < 0 else '#4C72B0' for x in df_subcat['Profit']]
sns.barplot(data=df_subcat, x='Sub_Category', y='Profit', palette=colors, ax=axes[1, 0])
axes[1, 0].axhline(0, color='black', linewidth=0.8)
axes[1, 0].set_title('3. Operational Leakage: Tables & Furniture Generate Deep Losses', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Net Profit ($)')

# Panel 4: Resolution / Correlation (Discount Rate vs Profitability)
discount_sample = pd.DataFrame({
    'Discount_Band': ['0-10%', '10-20%', '20-30%', '30-50%', '>50%'],
    'Avg_Margin': [28.5, 18.2, 4.1, -12.4, -38.6]
})
palette_line = ['#55A868' if x >= 0 else '#C44E52' for x in discount_sample['Avg_Margin']]
sns.barplot(data=discount_sample, x='Discount_Band', y='Avg_Margin', palette=palette_line, ax=axes[1, 1])
axes[1, 1].axhline(0, color='black', linewidth=0.8)
axes[1, 1].set_title('4. Structural Cliff: Discounts Above 20% Yield Compounded Losses', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Operating Margin (%)')

plt.tight_layout()
plt.show()