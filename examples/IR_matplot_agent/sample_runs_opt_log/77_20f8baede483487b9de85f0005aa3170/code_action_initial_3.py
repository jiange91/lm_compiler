import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse

# Load the Data
data = pd.read_csv('data.csv')

# Data Preparation
# Select relevant columns for clustering
features = data[["Red Meat", "White Meat", "Eggs", "Milk", "Fish", "Cereals", "Starch", "Nuts", "Fruits & Vegetables"]]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Plotting the Data
plt.figure(figsize=(12, 8))

# Define colors for the clusters
colors = ['red', 'green', 'blue']

# Plot each cluster
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Red Meat'], cluster_data['White Meat'], 
                color=colors[i], label=f'Cluster {i+1}', alpha=0.6)

    # Plot ellipses around clusters
    cov = np.cov(features_scaled[data['Cluster'] == i].T)
    mean = np.mean(features_scaled[data['Cluster'] == i], axis=0)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes of the ellipse
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.degrees(np.arctan2(u[1], u[0]))  # Correct angle calculation
    ell = Ellipse(mean[:2], v[0], v[1], angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Plot lines connecting points to centroids
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse transform to original scale
for i in range(len(data)):
    plt.plot([data.iloc[i]['Red Meat'], centroids[data.iloc[i]['Cluster'], 0]], 
             [data.iloc[i]['White Meat'], centroids[data.iloc[i]['Cluster'], 1]], 
             color='black', linestyle='--', alpha=0.5)

# Annotate points with country names
for i in range(len(data)):
    plt.annotate(data.iloc[i]['Country'], 
                 (data.iloc[i]['Red Meat'], data.iloc[i]['White Meat']),
                 textcoords="offset points", 
                 xytext=(0,5), 
                 ha='center')

# Set labels and title
plt.title('Protein Consumption Clustering in European Countries')
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.legend()
plt.grid()

# Save the plot to a PNG file
plt.savefig("novice.png")

# Show the plot
plt.show()