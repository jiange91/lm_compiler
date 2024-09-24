import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv("data.csv")

# Select features for clustering
features = data[["Red Meat", "White Meat"]]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['Red Meat'], data['White Meat'], c=data['Cluster'], alpha=0.6, cmap='viridis')

# Add annotations for each point
for i, txt in enumerate(data['Country']):
    plt.annotate(txt, (data['Red Meat'][i], data['White Meat'][i]), fontsize=9, alpha=0.7)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='X')

# Draw ellipses around clusters
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    ell = Ellipse(mean, v[0], v[1], angle=angle, color='gray', alpha=0.2)  # Corrected argument
    plt.gca().add_patch(ell)

# Labels and title
plt.title('Protein Consumption Clustering of European Countries')
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.legend()
plt.grid()

# Save the plot
plt.savefig("novice_final.png")
plt.show()