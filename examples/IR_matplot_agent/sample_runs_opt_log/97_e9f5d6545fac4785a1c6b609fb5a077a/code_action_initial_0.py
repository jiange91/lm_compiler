import pandas as pd
import matplotlib.pyplot as plt
import ternary
import numpy as np

# Load the dataset
data = pd.read_csv('data.csv')

# Extract relevant columns
data['toluene'] = data['toluene (25°C)']
data['n-heptane'] = data['n-heptane (25°C)']
data['IL'] = data['IL (25°C)']

# Normalize the data
data['total'] = data['toluene'] + data['n-heptane'] + data['IL']
data['toluene'] /= data['total']
data['n-heptane'] /= data['total']
data['IL'] /= data['total']

# Create the figure for the equilateral triangle diagram
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='ternary'))
ax.set_title("Liquid-Liquid Phase Diagram", fontsize=16)

# Plot each group with different colors
groups = data.groupby('No.')
colors = plt.cm.get_cmap('tab10', len(groups))

for i, (name, group) in enumerate(groups):
    ax.scatter(group['toluene'], group['n-heptane'], group['IL'], label=name, color=colors(i), s=100)
    ax.plot(group['toluene'], group['n-heptane'], linestyle='--', color=colors(i))

# Label the vertices
ax.set_tlabel("Toluene")
ax.set_llabel("n-Heptane")
ax.set_rlabel("IL")
ax.boundary.set_color('black')
ax.clear_ticks()
ax.legend(title='Group No.')

# Save the equilateral triangle diagram
plt.savefig('novice_equilateral.png')

# Create the figure for the right-angled triangle diagram
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_title("Liquid-Liquid Phase Diagram", fontsize=16)

# Plot each group with different colors
for i, (name, group) in enumerate(groups):
    ax2.scatter(group['IL'], group['toluene'], label=name, color=colors(i), s=100)
    ax2.plot(group['IL'], group['toluene'], linestyle='--', color=colors(i))

# Label the axes
ax2.set_ylabel("Toluene")
ax2.set_xlabel("IL")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('black')
ax2.spines['bottom'].set_color('black')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.legend(title='Group No.')

# Save the right-angled triangle diagram
plt.savefig('novice_rightangled.png')

# Show both diagrams
plt.show()