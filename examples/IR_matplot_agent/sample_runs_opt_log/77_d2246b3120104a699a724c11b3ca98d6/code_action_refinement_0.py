import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

# Step 1: Load the Data
data = pd.read_csv("data.csv")

# Step 2: Prepare the Data for Clustering
X = data[["Red Meat", "White Meat"]]
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)
data['Centroid'] = kmeans.cluster_centers_[data['Cluster']]

# Step 3: Create the Scatter Plot
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i}', alpha=0.5)

# Plotting centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

# Adding ellipses around clusters
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    mean = np.mean(cluster_data[['Red Meat', 'White Meat']], axis=0)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    ell = Ellipse(mean, v[0], v[1], angle=np.degrees(np.arctan2(w[0][1], w[0][0])), color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Connecting points to centroids
for i in range(len(data)):
    plt.plot([data['Red Meat'][i], centroids[data['Cluster'][i], 0]], 
             [data['White Meat'][i], centroids[data['Cluster'][i], 1]], 
             color='gray', linestyle='--', alpha=0.5)

# Annotating points with country names
for i in range(len(data)):
    plt.annotate(data['Country'][i], (data['Red Meat'][i], data['White Meat'][i]), fontsize=8)

# Labels and title
plt.title('Protein Consumption Clustering in European Countries')
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.legend()
plt.grid()

# Step 4: Save the Plot
plt.savefig("novice_final.png")
plt.show()