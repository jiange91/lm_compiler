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
sns.set(style="whitegrid")

# Scatter plot
scatter = plt.scatter(data['Red Meat'], data['White Meat'], c=data['Cluster'], cmap='viridis', alpha=0.6, edgecolor='k')

# Add annotations for each point
for i, country in enumerate(data['Country']):
    plt.annotate(country, (data['Red Meat'][i], data['White Meat'][i]), fontsize=9, alpha=0.7)

# Function to draw ellipses around clusters
def draw_ellipse(ax, mean, cov, color):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale by 2 standard deviations
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # Convert to degrees
    ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color=color, alpha=0.2)
    ax.add_patch(ell)

# Draw ellipses for each cluster
for cluster in range(3):
    cluster_data = features_scaled[data['Cluster'] == cluster]
    mean = np.mean(cluster_data, axis=0)
    cov = np.cov(cluster_data, rowvar=False)
    draw_ellipse(plt.gca(), mean, cov, color=scatter.cmap(cluster / 3))

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

# Add labels and legend
plt.title('Protein Consumption Clustering of European Countries', fontsize=16)
plt.xlabel('Red Meat Consumption', fontsize=14)
plt.ylabel('White Meat Consumption', fontsize=14)
plt.legend(loc='upper right')
plt.grid()

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()