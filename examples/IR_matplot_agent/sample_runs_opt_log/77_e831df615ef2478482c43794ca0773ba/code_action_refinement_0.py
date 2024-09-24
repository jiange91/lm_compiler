import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import numpy as np

# Load the Data
data = pd.read_csv("data.csv")

# Prepare Data for Clustering
X = data[["Red Meat", "White Meat"]]

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Plot the Data
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i + 1}', alpha=0.6)

# Draw Ellipses Around Clusters
def draw_ellipse(cluster_data, color):
    cov = np.cov(cluster_data.T)
    mean = np.mean(cluster_data, axis=0)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale for ellipse
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    ellipse = Ellipse(mean, v[0], v[1], angle=np.degrees(angle), color=color, alpha=0.2)
    plt.gca().add_patch(ellipse)

for i in range(3):
    draw_ellipse(data[data['Cluster'] == i][["Red Meat", "White Meat"]].values, colors[i])

# Connect Points to Centroids
centroids = kmeans.cluster_centers_
for i in range(len(data)):
    plt.plot([data['Red Meat'].iloc[i], centroids[data['Cluster'].iloc[i], 0]], 
             [data['White Meat'].iloc[i], centroids[data['Cluster'].iloc[i], 1]], 
             color='black', linestyle='--', alpha=0.5)

# Annotate Points
for i in range(len(data)):
    plt.annotate(data['Country'].iloc[i], (data['Red Meat'].iloc[i], data['White Meat'].iloc[i]), 
                 fontsize=8, alpha=0.7)

# Add Labels and Legend
plt.title("Protein Consumption Clustering in European Countries")
plt.xlabel("Red Meat Consumption")
plt.ylabel("White Meat Consumption")
plt.legend(title="Clusters")

# Save the Plot
plt.savefig("novice_final.png", bbox_inches='tight')
plt.show()