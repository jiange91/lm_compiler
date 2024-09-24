# Step 1: Install Required Libraries
# Make sure to run this in your terminal or command prompt
# pip install pandas matplotlib seaborn scikit-learn

# Step 2: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import numpy as np

# Step 3: Load the Data
data = pd.read_csv('data.csv')

# Step 4: Data Preparation
# Select relevant columns for clustering
features = data[['Red Meat', 'White Meat', 'Eggs', 'Milk', 'Fish', 'Cereals', 'Starch', 'Nuts', 'Fruits & Vegetables']]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Get cluster centroids
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse transform to original scale

# Step 6: Create the Scatter Plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
sns.set_palette(sns.color_palette(colors))

# Scatter plot
plt.scatter(data['Red Meat'], data['White Meat'], c=data['Cluster'], alpha=0.6, edgecolor='k')

# Annotate each point with the country name
for i, row in data.iterrows():
    plt.annotate(row['Country'], (row['Red Meat'], row['White Meat']), fontsize=9, alpha=0.7)

# Plot ellipses around clusters
for i in range(3):
    cluster_data = scaled_features[data['Cluster'] == i]
    mean = np.mean(cluster_data, axis=0)
    cov = np.cov(cluster_data, rowvar=False)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # Convert to degrees
    ell = Ellipse(mean, v[0], v[1], angle=180 + angle, color=colors[i], alpha=0.2)  # Corrected here
    plt.gca().add_patch(ell)

# Draw lines connecting points to centroids
for i, centroid in enumerate(centroids):
    plt.plot([centroid[0], centroid[0]], [centroid[1], centroid[1]], 'k--', alpha=0.5)

# Set labels and title
plt.title('Protein Consumption Clustering in European Countries', fontsize=16)
plt.xlabel('Red Meat Consumption', fontsize=14)
plt.ylabel('White Meat Consumption', fontsize=14)
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', markerfacecolor='red', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2', markerfacecolor='green', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Cluster 3', markerfacecolor='blue', markersize=10)],
           title='Clusters', loc='upper right')

# Save the plot to a PNG file
plt.grid()
plt.savefig("novice.png")

# Show the plot
plt.show()