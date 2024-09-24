import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import numpy as np

# Load the data from CSV
data = pd.read_csv('data.csv')

# Select the columns for clustering
features = data[['Red Meat', 'White Meat', 'Eggs', 'Milk', 'Fish', 'Cereals', 'Starch', 'Nuts', 'Fruits & Vegetables']]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Get the cluster centroids
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse transform to original scale

# Set up the plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
sns.set_palette(sns.color_palette(colors))

# Scatter plot
plt.scatter(data['Red Meat'], data['White Meat'], c=data['Cluster'], s=100, alpha=0.6, edgecolor='k')

# Annotate each point with the country name
for i, row in data.iterrows():
    plt.annotate(row['Country'], (row['Red Meat'], row['White Meat']), fontsize=9, alpha=0.7)

# Draw ellipses around clusters
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].values.T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    
    # Create an ellipse
    ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Draw lines connecting points to centroids
for i, row in data.iterrows():
    plt.plot([row['Red Meat'], centroids[row['Cluster'], 0]], 
             [row['White Meat'], centroids[row['Cluster'], 1]], 
             color='black', linestyle='--', alpha=0.5)

# Set labels and title
plt.title('Protein Consumption Clustering in European Countries', fontsize=16)
plt.xlabel('Red Meat Consumption', fontsize=14)
plt.ylabel('White Meat Consumption', fontsize=14)
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', 
                                  markerfacecolor='red', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2', 
                                  markerfacecolor='green', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 3', 
                                  markerfacecolor='blue', markersize=10)],
           title='Clusters', loc='upper right')

# Show the plot
plt.grid()
plt.tight_layout()
plt.savefig('novice.png')
plt.show()