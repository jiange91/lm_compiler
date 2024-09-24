import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

# Step 1: Load the Data
data = pd.read_csv("data.csv")

# Step 2: Prepare the Data
X = data[["Red Meat", "White Meat"]]

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Step 4: Create the Scatter Plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], color=colors[i], label=f'Cluster {i + 1}', alpha=0.6)

# Step 5: Add Ellipses Around Clusters
def draw_ellipse(cluster_data, color):
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize
    angle = np.arctan(u[1] / u[0]) * 180.0 / np.pi
    ell = Ellipse(mean, v[0], v[1], angle, color=color, alpha=0.2)
    plt.gca().add_patch(ell)

for i in range(3):
    draw_ellipse(data[data['Cluster'] == i], colors[i])

# Step 6: Connect Points to Centroids
centroids = kmeans.cluster_centers_
for i in range(len(data)):
    plt.plot([data['Red Meat'].iloc[i], centroids[data['Cluster'].iloc[i], 0]],
             [data['White Meat'].iloc[i], centroids[data['Cluster'].iloc[i], 1]], 
             color='black', linestyle='--', alpha=0.5)

# Step 7: Annotate Points
for i in range(len(data)):
    plt.annotate(data['Country'].iloc[i], (data['Red Meat'].iloc[i], data['White Meat'].iloc[i]), fontsize=9, ha='right')

# Step 8: Add Labels and Legend
plt.title("Protein Consumption Clustering in European Countries")
plt.xlabel("Red Meat Consumption")
plt.ylabel("White Meat Consumption")
plt.legend(title="Clusters")
plt.grid()

# Step 9: Save the Plot
plt.savefig("novice_final.png")
plt.show()