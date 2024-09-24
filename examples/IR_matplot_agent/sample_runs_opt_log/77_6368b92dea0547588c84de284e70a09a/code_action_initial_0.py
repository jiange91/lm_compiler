import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import numpy as np

# Load the Data
data = pd.read_csv('data.csv')

# Data Preparation
# Select relevant columns for clustering
features = data[['Red Meat', 'White Meat', 'Eggs', 'Milk', 'Fish', 'Cereals', 'Starch', 'Nuts', 'Fruits & Vegetables']]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Get cluster centroids
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse transform to original scale

# Create the Scatter Plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
sns.set_palette(sns.color_palette(colors))

# Scatter plot
plt.scatter(data['Red Meat'], data['White Meat'], c=data['Cluster'], alpha=0.6, edgecolor='k')

# Add ellipses around clusters
for i in range(3):
    cluster_data = scaled_features[data['Cluster'] == i]
    cov = np.cov(cluster_data.T)
    mean = np.mean(cluster_data, axis=0)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    ell = Ellipse(mean, v[0], v[1], angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')

# Connect points to centroids
for i in range(len(data)):
    plt.plot([data['Red Meat'][i], centroids[data['Cluster'][i], 0]], 
             [data['White Meat'][i], centroids[data['Cluster'][i], 1]], 
             color='gray', linestyle='--', alpha=0.5)

# Annotate points with country names
for i in range(len(data)):
    plt.annotate(data['Country'][i], (data['Red Meat'][i], data['White Meat'][i]), fontsize=8, alpha=0.7)

# Labeling the plot
plt.title('Protein Consumption Clustering in European Countries', fontsize=16)
plt.xlabel('Red Meat Consumption', fontsize=14)
plt.ylabel('White Meat Consumption', fontsize=14)
plt.legend()
plt.grid()

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()