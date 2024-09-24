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
colors = ['yellow', 'red', 'green']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# Scatter plot
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=labels[i], alpha=0.6)

# Draw ellipses around clusters
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0]) * 180.0 / np.pi
    ell = Ellipse(mean, v[0], v[1], angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Draw lines connecting points to centroids
centroids = kmeans.cluster_centers_
for i in range(len(data)):
    plt.plot([data['Red Meat'][i], centroids[data['Cluster'][i]][0]], 
             [data['White Meat'][i], centroids[data['Cluster'][i]][1]], 
             color='gray', linestyle='--', alpha=0.5)

# Annotations
for i in range(len(data)):
    plt.annotate(data['Country'][i], (data['Red Meat'][i], data['White Meat'][i]), 
                 fontsize=8, ha='right')

# Labels and title
plt.title("Protein Consumption Clustering in European Countries", fontsize=16)
plt.xlabel("Red Meat Consumption", fontsize=14)
plt.ylabel("White Meat Consumption", fontsize=14)
plt.legend()
plt.grid()

# Save the plot
plt.savefig("novice_final.png")
plt.show()