# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Step 3: Data Preparation
# Select relevant columns for clustering
features = data[["Red Meat", "White Meat", "Eggs", "Milk", "Fish", "Cereals", "Starch", "Nuts", "Fruits & Vegetables"]]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 4: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Step 5: Plotting the Data
plt.figure(figsize=(12, 8))

# Define colors for the clusters
colors = ['red', 'green', 'blue']

# Plot each cluster
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i+1}', alpha=0.6)

    # Plot ellipses around clusters
    cov = np.cov(features_scaled[data['Cluster'] == i].T)
    mean = np.mean(features_scaled[data['Cluster'] == i], axis=0)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes of the ellipse
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    ell = Ellipse(xy=mean[:2], width=v[0], height=v[1], angle=angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Plot centroids
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse transform to original scale
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

# Annotate each point with the country name
for i, row in data.iterrows():
    plt.annotate(row['Country'], (row['Red Meat'], row['White Meat']), fontsize=8, alpha=0.7)

# Add labels and legend
plt.title('Protein Consumption Clustering in European Countries')
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.legend()
plt.grid()

# Save the plot to a file
plt.savefig('novice.png')
plt.show()