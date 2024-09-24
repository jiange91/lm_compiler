import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import numpy as np

# Load the Data
data = pd.read_csv("data.csv")
X = data[["Red Meat", "White Meat"]]

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Create the Scatter Plot
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
for i in range(3):
    plt.scatter(X[data['Cluster'] == i]['Red Meat'], 
                X[data['Cluster'] == i]['White Meat'], 
                color=colors[i], label=f'Cluster {i+1}', alpha=0.6)

# Add Ellipses Around Clusters
def draw_ellipse(ax, mean, cov, color):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale by 2 for visibility
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # Convert to degrees
    ell = Ellipse(mean, v[0], v[1], angle=angle, color=color, alpha=0.2)  # Fixed here
    ax.add_patch(ell)

for i in range(3):
    mean = np.mean(X[data['Cluster'] == i], axis=0)
    cov = np.cov(X[data['Cluster'] == i].T)
    draw_ellipse(plt.gca(), mean, cov, colors[i])

# Connect Points to Centroids
for i in range(len(X)):
    plt.plot([X.iloc[i]['Red Meat'], centroids[data.iloc[i]['Cluster'], 0]], 
             [X.iloc[i]['White Meat'], centroids[data.iloc[i]['Cluster'], 1]], 
             color='gray', linestyle='--', alpha=0.5)

# Annotate Points
for i in range(len(data)):
    plt.annotate(data['Country'][i], (X.iloc[i]['Red Meat'], X.iloc[i]['White Meat']), 
                 fontsize=8, alpha=0.7)

# Add Labels and Title
plt.title("Protein Consumption Clustering in European Countries")
plt.xlabel("Red Meat Consumption")
plt.ylabel("White Meat Consumption")
plt.legend()

# Save the Plot
plt.savefig("novice_final.png", bbox_inches='tight')
plt.show()