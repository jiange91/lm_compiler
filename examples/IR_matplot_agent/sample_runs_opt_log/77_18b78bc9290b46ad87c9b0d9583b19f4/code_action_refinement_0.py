import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Load the data
data = pd.read_csv("data.csv")

# Select relevant columns for clustering
features = data[["Red Meat", "White Meat"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Create scatter plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[cluster], label=f'Cluster {cluster + 1}', alpha=0.6)

    # Calculate the mean and covariance for the ellipse
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector

    # Create the ellipse
    angle = np.arctan(u[1] / u[0])
    ellipse = Ellipse(mean, v[0], v[1], angle=np.degrees(angle), color=colors[cluster], alpha=0.2)
    plt.gca().add_patch(ellipse)

    # Connect points to centroids
    centroid = kmeans.cluster_centers_[cluster]
    for i in cluster_data.index:
        plt.plot([data.loc[i, 'Red Meat'], centroid[0]], 
                 [data.loc[i, 'White Meat'], centroid[1]], color='black', linestyle='--', alpha=0.5)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='black', marker='X', s=200, label='Centroids')

# Annotations for each point
for i in data.index:
    plt.annotate(data['Country'][i], (data['Red Meat'][i], data['White Meat'][i]), 
                 fontsize=8, alpha=0.7)

# Labels and title
plt.title("Protein Consumption Clusters in European Countries")
plt.xlabel("Red Meat Consumption (Standardized)")
plt.ylabel("White Meat Consumption (Standardized)")
plt.legend()
plt.grid()

# Save the plot
plt.savefig("novice_final.png")
plt.show()