import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse

# Load the data
data = pd.read_csv("data.csv")

# Select relevant columns for clustering
X = data[["Red Meat", "White Meat"]]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Plotting
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[cluster], label=f'Cluster {cluster + 1}', alpha=0.6)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

# Draw ellipses around clusters
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    mean = np.mean(cluster_data[['Red Meat', 'White Meat']], axis=0)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    ell = Ellipse(mean, v[0], v[1], angle, color=colors[cluster], alpha=0.2)
    plt.gca().add_patch(ell)

# Annotate points with country names
for i, row in data.iterrows():
    plt.annotate(row['Country'], (row['Red Meat'], row['White Meat']), fontsize=8)

# Labels and title
plt.title('Protein Consumption Clustering of European Countries')
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.legend()
plt.grid()

# Save the plot
plt.savefig("novice_final.png")
plt.show()