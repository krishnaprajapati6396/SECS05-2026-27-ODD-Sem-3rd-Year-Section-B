import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# ==============================================================================
# Step 1 & 2: Load the Dataset and Inspect
# ==============================================================================
filename = "netflix_titles.csv"
df = pd.read_csv(filename)
print(f"Dataset successfully loaded. Shape: {df.shape}\n")

# Preprocessing: Extract numeric movie duration and release year
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['year_added'] = df['date_added'].dt.year

# Extract duration integer for movies
movies_df = df[df['type'] == 'Movie'].copy()
movies_df['duration_min'] = movies_df['duration'].str.extract('(\d+)').astype(float)

# ==============================================================================
# Step 3: Bar Chart - Content Type Distribution (Movies vs TV Shows)
# ==============================================================================
plt.figure(figsize=(7, 5))
type_counts = df['type'].value_counts()
sns.barplot(x=type_counts.index, y=type_counts.values, palette='Blues_d')
plt.title("Distribution of Content Type on Netflix", fontsize=14, weight='bold')
plt.xlabel("Content Type", fontsize=12)
plt.ylabel("Total Count", fontsize=12)
plt.savefig("1_bar_chart_content_type.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] 1_bar_chart_content_type.png")

# ==============================================================================
# Step 4: Line Chart - Releases Added Per Year Over Time
# ==============================================================================
plt.figure(figsize=(10, 5))
yearly_trend = df.groupby(['year_added', 'type']).size().unstack().fillna(0)
# Filter realistic release years for Netflix catalog additions
yearly_trend = yearly_trend[yearly_trend.index >= 2010]

plt.plot(yearly_trend.index, yearly_trend['Movie'], marker='o', label='Movies', color='#e50914', linewidth=2)
plt.plot(yearly_trend.index, yearly_trend['TV Show'], marker='s', label='TV Shows', color='#221f1f', linewidth=2)
plt.title("Netflix Catalog Growth Over Time (2010 - Present)", fontsize=14, weight='bold')
plt.xlabel("Year Added", fontsize=12)
plt.ylabel("Number of Titles Added", fontsize=12)
plt.legend()
plt.savefig("2_line_chart_yearly_trend.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] 2_line_chart_yearly_trend.png")

# ==============================================================================
# Step 5: Histogram - Distribution of Movie Durations (in Minutes)
# ==============================================================================
plt.figure(figsize=(8, 5))
sns.histplot(movies_df['duration_min'].dropna(), bins=30, kde=True, color='#e50914')
plt.title("Distribution of Movie Durations on Netflix", fontsize=14, weight='bold')
plt.xlabel("Duration (Minutes)", fontsize=12)
plt.ylabel("Number of Movies", fontsize=12)
plt.savefig("3_histogram_movie_duration.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] 3_histogram_movie_duration.png")

# ==============================================================================
# Step 6: Box Plot - Movie Durations across Top Ratings
# ==============================================================================
plt.figure(figsize=(9, 5))
top_ratings = movies_df['rating'].value_counts().index[:6]
filtered_ratings = movies_df[movies_df['rating'].isin(top_ratings)]

sns.boxplot(data=filtered_ratings, x='rating', y='duration_min', palette='Set2')
plt.title("Movie Duration Distribution Across Top Content Ratings", fontsize=14, weight='bold')
plt.xlabel("Content Rating", fontsize=12)
plt.ylabel("Duration (Minutes)", fontsize=12)
plt.savefig("4_boxplot_duration_by_rating.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] 4_boxplot_duration_by_rating.png")

# ==============================================================================
# Step 7: Scatter Plot - Release Year vs Movie Duration
# ==============================================================================
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=movies_df[movies_df['release_year'] >= 1990], 
    x='release_year', 
    y='duration_min', 
    alpha=0.4, 
    color='#e50914'
)
plt.title("Release Year vs. Movie Duration (1990 - Present)", fontsize=14, weight='bold')
plt.xlabel("Release Year", fontsize=12)
plt.ylabel("Duration (Minutes)", fontsize=12)
plt.savefig("5_scatterplot_year_vs_duration.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] 5_scatterplot_year_vs_duration.png")

# ==============================================================================
# Step 8: Correlation Heatmap of Numerical Features
# ==============================================================================
plt.figure(figsize=(7, 5))
num_df = df[['release_year']].copy()
num_df['year_added'] = df['year_added']
num_df['duration_min'] = movies_df['duration_min']

corr = num_df.corr(method='pearson')
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Numeric Netflix Attributes", fontsize=14, weight='bold')
plt.savefig("6_heatmap_correlation.png", dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] 6_heatmap_correlation.png")

print("\nAll 6 experiment plots generated and saved as high-res PNG files.")

