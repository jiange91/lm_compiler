import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv("data.csv")

# Select relevant features
X = data[["Red Meat", "White Meat"]]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Create the scatter plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i + 1}', alpha=0.6)

    # Calculate the mean for the cluster
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    plt.scatter(mean[0], mean[1], color='black', marker='X', s=200)  # Centroid

    # Draw ellipses around clusters
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # Convert to degrees
    ell = Ellipse(mean, v[0], v[1], angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

    # Connect points to centroids
    for index, row in cluster_data.iterrows():
        plt.plot([row['Red Meat'], mean[0]], [row['White Meat'], mean[1]], 
                 color='gray', linestyle='--', alpha=0.5)

# Annotate points with country names
for i, row in data.iterrows():
    plt.annotate(row['Country'], (row['Red Meat'], row['White Meat']), 
                 textcoords="offset points", xytext=(0,10), ha='center')

# Set labels and title
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.title('Protein Consumption Clustering of European Countries')
plt.legend()
plt.grid()

# Save the plot
plt.savefig("novice_final.png")
plt.show()