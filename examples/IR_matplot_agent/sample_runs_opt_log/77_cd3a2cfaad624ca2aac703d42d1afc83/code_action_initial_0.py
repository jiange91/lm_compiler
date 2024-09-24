import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Select relevant columns for clustering
features = data[['Red Meat', 'White Meat', 'Eggs', 'Milk', 'Fish', 'Cereals', 'Starch', 'Nuts', 'Fruits & Vegetables']]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Get cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Set up the plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
markers = ['o', 's', 'D']

# Plot each cluster
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i+1}', alpha=0.6, edgecolor='k', s=100)

    # Draw ellipses around clusters
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=np.arctan2(w[0][1], w[0][0]) * 180. / np.pi,
                   color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

    # Connect points to centroids
    for index, row in cluster_data.iterrows():
        plt.plot([row['Red Meat'], centroids[i][0]], [row['White Meat'], centroids[i][1]], color='gray', linestyle='--')

# Annotate points with country names
for index, row in data.iterrows():
    plt.annotate(row['Country'], (row['Red Meat'], row['White Meat']), fontsize=9, ha='right')

# Labeling the plot
plt.title('Protein Consumption Clustering of European Countries', fontsize=16)
plt.xlabel('Red Meat Consumption', fontsize=14)
plt.ylabel('White Meat Consumption', fontsize=14)
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()