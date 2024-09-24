import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data from CSV file
data = pd.read_csv('data.csv')

# Select relevant columns for clustering
X = data[["Red Meat", "White Meat", "Eggs", "Milk", "Fish", "Cereals", "Starch", "Nuts", "Fruits & Vegetables"]]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Create a scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='Red Meat', y='White Meat', hue='Cluster', palette='Set1', s=100)

# Draw ellipses around clusters
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    mean = cluster_data[['Red Meat', 'White Meat']].mean().values
    cov = np.cov(cluster_data[['Red Meat', 'White Meat']].T)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    
    # Corrected Ellipse initialization
    ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], angle, color='gray', alpha=0.2)
    plt.gca().add_patch(ell)

# Draw lines connecting data points to centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
for i in range(len(data)):
    plt.plot([data['Red Meat'][i], centroids[data['Cluster'][i], 0]], 
             [data['White Meat'][i], centroids[data['Cluster'][i], 1]], 
             color='black', linestyle='--', alpha=0.5)

# Annotate each point with the country name
for i in range(len(data)):
    plt.text(data['Red Meat'][i], data['White Meat'][i], data['Country'][i], 
             horizontalalignment='right', size='medium', color='black', weight='semibold')

# Add labels and legend
plt.title('Protein Consumption Clustering in European Countries')
plt.xlabel('Red Meat Consumption')
plt.ylabel('White Meat Consumption')
plt.legend(title='Cluster', loc='upper right')

# Save the plot to a PNG file
plt.grid()
plt.tight_layout()
plt.savefig('novice.png')

# Show the plot
plt.show()