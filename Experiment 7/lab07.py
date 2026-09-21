import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 0. Setup Output Directory
# ==========================================
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. Load Dataset
# ==========================================
DATA_URL = "https://raw.githubusercontent.com/sharmaroshan/Clustering-of-Mall-Customers/master/Mall_Customers.csv"
df = pd.read_csv(DATA_URL)

# Clean column headers
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Select features & scale
feature_cols = ['Annual_Income_k$', 'Spending_Score_1-100']
X = df[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. Elbow & Silhouette Curves
# ==========================================
wcss = []
k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Curve
axes[0].plot(list(k_values), wcss, marker='o', color='#1f77b4', linewidth=2)
axes[0].set_title('Elbow Method (WCSS vs. K)')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Silhouette Curve
axes[1].plot(list(k_values), silhouette_scores, marker='s', color='#2ca02c', linewidth=2)
axes[1].set_title('Silhouette Score vs. K')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Save Image 1
fig.savefig(os.path.join(output_dir, "elbow_silhouette_analysis.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'elbow_silhouette_analysis.png')}")
plt.show()

# ==========================================
# 3. Fit Model & Plot Clusters
# ==========================================
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(9, 6))
palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

sns.scatterplot(
    data=df, 
    x='Annual_Income_k$', 
    y='Spending_Score_1-100', 
    hue='Cluster', 
    palette=palette, 
    s=70, 
    alpha=0.9
)

# Centroids
plt.scatter(
    centroids_original[:, 0], 
    centroids_original[:, 1], 
    s=250, 
    c='black', 
    marker='X', 
    edgecolor='white', 
    linewidth=1.5, 
    label='Centroids'
)

plt.title('Customer Segments (K-Means Clustering, K=5)', fontsize=13)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save Image 2
plt.savefig(os.path.join(output_dir, "customer_segments_clusters.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'customer_segments_clusters.png')}")
plt.show()