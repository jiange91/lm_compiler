# Step 1: Install Required Libraries
# Make sure to run this in your terminal or command prompt if you haven't installed the libraries yet.
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

# Display the first few rows of the data
print(data.head())

# Step 4: Data Preparation
# Select the features for clustering
features = data[['Red Meat', 'White Meat', 'Eggs', 'Milk', 'Fish', 'Cereals', 'Starch', 'Nuts', 'Fruits & Vegetables']]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Get the cluster centroids
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse transform to original scale

# Step 6: Create the Scatter Plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
sns.set_palette(sns.color_palette(colors))

# Scatter plot
sns.scatterplot(data=data, x='Red Meat', y='White Meat', hue='Cluster', style='Cluster', s=100, alpha=0.7)

# Add annotations for each point
for i in range(data.shape[0]):
    plt.text(data['Red Meat'].iloc[i], data['White Meat'].iloc[i], data['Country'].iloc[i], 
             horizontalalignment='right', size='medium', color='black', weight='semibold')

# Draw ellipses around clusters
for i in range(3):
    cluster_data = scaled_features[data['Cluster'] == i]
    mean = np.mean(cluster_data, axis=0)
    cov = np.cov(cluster_data, rowvar=False)
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes of the ellipse
    u = w[0] / np.linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.arctan(u[1] / u[0])
    angle = np.degrees(angle)
    
    # Create the ellipse
    ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color=colors[i], alpha=0.2)
    plt.gca().add_patch(ell)

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

# Add labels and title
plt.title('Protein Consumption Clustering in European Countries', fontsize=16)
plt.xlabel('Red Meat Consumption', fontsize=14)
plt.ylabel('White Meat Consumption', fontsize=14)
plt.legend(title='Cluster', loc='upper right')
plt.grid()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()