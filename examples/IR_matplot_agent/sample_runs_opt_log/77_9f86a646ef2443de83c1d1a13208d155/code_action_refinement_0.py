import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
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
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i}', alpha=0.6)

    # Step 5: Draw ellipses around clusters
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # Convert to degrees
    ell = Ellipse(mean, v[0], v[1], 180 + angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Step 6: Connect points to centroids
centroids = kmeans.cluster_centers_
for i in range(len(data)):
    plt.plot([data['Red Meat'][i], centroids[data['Cluster'][i]][0]], 
             [data['White Meat'][i], centroids[data['Cluster'][i]][1]], 
             color='gray', linestyle='--', alpha=0.5)

# Step 7: Annotate points with country names
for i in range(len(data)):
    plt.annotate(data['Country'][i], (data['Red Meat'][i], data['White Meat'][i]), 
                 textcoords="offset points", xytext=(0,5), ha='center')

# Step 8: Labeling and legend
plt.title("Protein Consumption Clustering of European Countries")
plt.xlabel("Red Meat Consumption")
plt.ylabel("White Meat Consumption")
plt.legend()
plt.grid()

# Step 9: Save the plot
plt.savefig("novice_final.png")
plt.show()