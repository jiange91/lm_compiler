# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import numpy as np

# Step 2: Load the Data
data = pd.read_csv('data.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Step 3: Prepare the Data
# Select the columns for clustering
features = data[["Red Meat", "White Meat", "Eggs", "Milk", "Fish", "Cereals", "Starch", "Nuts", "Fruits & Vegetables"]]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 4: Apply K-Means Clustering
# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Get the cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Step 5: Create a Scatter Plot with Clusters
# Set up the plot
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']
sns.set_palette(sns.color_palette(colors))

# Scatter plot
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=data['Cluster'], alpha=0.6, edgecolor='k')

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centroids')

# Annotate each point with the country name
for i, txt in enumerate(data['Country']):
    plt.annotate(txt, (features_scaled[i, 0], features_scaled[i, 1]), fontsize=9, alpha=0.7)

# Draw ellipses around clusters
def draw_ellipse(ax, mean, cov, color):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # Convert to degrees
    ell = Ellipse(mean, v[0], v[1], 180 + angle, color=color, alpha=0.2)
    ax.add_patch(ell)

# Draw ellipses for each cluster
for i in range(3):
    cluster_data = features_scaled[data['Cluster'] == i]
    mean = np.mean(cluster_data, axis=0)
    cov = np.cov(cluster_data, rowvar=False)
    draw_ellipse(plt.gca(), mean, cov, colors[i])

# Finalize the plot
plt.title('Protein Consumption Clusters in European Countries')
plt.xlabel('Red Meat Consumption (Standardized)')
plt.ylabel('White Meat Consumption (Standardized)')
plt.legend()
plt.grid()

# Save the plot to a file
plt.savefig('novice.png')

# Show the plot
plt.show()