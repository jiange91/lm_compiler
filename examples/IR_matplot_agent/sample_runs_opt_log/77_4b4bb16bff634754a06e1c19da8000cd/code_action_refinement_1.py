import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans

# Load the Data
data = pd.read_csv("data.csv")

# Select Features for Clustering
X = data[["Red Meat", "White Meat"]]

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Plotting the Data
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i + 1}', alpha=0.6)
    
    # Calculate the ellipse parameters
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    angle = np.arctan2(w[0][1], w[0][0])
    angle = np.degrees(angle)
    
    # Correctly set the center of the ellipse
    ellipse_center = cluster_data[['Red Meat', 'White Meat']].mean().values
    ell = Ellipse(xy=ellipse_center, width=v[0], height=v[1],
                   angle=angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=200, label='Centroids')

# Connecting Points to Centroids
for i in range(len(data)):
    plt.plot([data['Red Meat'].iloc[i], centroids[data['Cluster'].iloc[i], 0]],
             [data['White Meat'].iloc[i], centroids[data['Cluster'].iloc[i], 1]], 
             color='gray', linestyle='--', alpha=0.5)

# Annotate Points
for i in range(len(data)):
    plt.annotate(data['Country'].iloc[i], 
                 (data['Red Meat'].iloc[i], data['White Meat'].iloc[i]), 
                 textcoords="offset points", 
                 xytext=(0,5), 
                 ha='center')

# Labeling and Legend
plt.title("Protein Consumption Clustering in European Countries")
plt.xlabel("Red Meat Consumption")
plt.ylabel("White Meat Consumption")
plt.legend()

# Save the Plot
plt.savefig("novice_final.png")
plt.show()