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
features = data[["Red Meat", "White Meat", "Eggs", "Milk", "Fish", "Cereals", "Starch", "Nuts", "Fruits & Vegetables"]]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Get cluster centroids
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse transform to original scale

# Set up the plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']

# Scatter plot
sns.scatterplot(data=data, x='Red Meat', y='White Meat', hue='Cluster', palette=colors, s=100, alpha=0.7)

# Add ellipses around clusters
for i in range(3):
    cluster_data = scaled_features[data['Cluster'] == i]
    mean = np.mean(cluster_data, axis=0)
    cov = np.cov(cluster_data, rowvar=False)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # Convert to degrees
    ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=180 + angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Draw lines connecting points to centroids
for i in range(len(data)):
    plt.plot([data['Red Meat'][i], centroids[data['Cluster'][i], 0]], 
             [data['White Meat'][i], centroids[data['Cluster'][i], 1]], 
             color='black', linestyle='--', alpha=0.5)

# Annotate points with country names
for i in range(len(data)):
    plt.text(data['Red Meat'][i], data['White Meat'][i], data['Country'][i], 
             horizontalalignment='right', size='medium', color='black', weight='semibold')

# Set labels and title
plt.title('Protein Consumption Clustering of European Countries')
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.legend(title='Cluster', loc='upper right')
plt.grid()

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()