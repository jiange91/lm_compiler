import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.patches import Ellipse

# Step 1: Load the data
data = pd.read_csv("data.csv")

# Step 2: Prepare the data for clustering
X = data[["Red Meat", "White Meat"]]

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Step 4: Plotting
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i + 1}', alpha=0.6)

# Step 5: Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=200, label='Centroids')

# Step 6: Draw ellipses around clusters
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    mean = np.mean(cluster_data[['Red Meat', 'White Meat']], axis=0)
    
    # Ensure covariance matrix is not singular
    if np.linalg.det(cov) > 0:
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
        u = w[0] / np.linalg.norm(w[0])  # Normalize
        angle = np.arctan(u[1] / u[0]) * 180.0 / np.pi
        
        # Create the ellipse
        ell = Ellipse(mean, v[0], v[1], angle, color=colors[i], alpha=0.2)
        plt.gca().add_patch(ell)

# Step 7: Annotate points with country names
for i, row in data.iterrows():
    plt.annotate(row['Country'], (row['Red Meat'], row['White Meat']), fontsize=8)

# Step 8: Labeling and legend
plt.title("Protein Consumption Clustering in European Countries")
plt.xlabel("Red Meat Consumption")
plt.ylabel("White Meat Consumption")
plt.legend()
plt.grid()

# Step 9: Save the plot
plt.savefig("novice_final.png")
plt.show()