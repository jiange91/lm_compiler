import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ternary

# Load the dataset
data = pd.read_csv('data.csv')

# Extract relevant columns
data['IL'] = data['IL (25°C)']
data['Toluene'] = data['toluene (25°C)']
data['N-Heptane'] = data['n-heptane (25°C)']

# Normalize the data to get fractions
data['Total'] = data['IL'] + data['Toluene'] + data['N-Heptane']
data['IL_frac'] = data['IL'] / data['Total']
data['Toluene_frac'] = data['Toluene'] / data['Total']
data['N-Heptane_frac'] = data['N-Heptane'] / data['Total']

# Create the equilateral triangle plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='ternary'))
ax.set_title("Liquid-Liquid Phase Diagram", fontsize=16)
groups = data.groupby('No.')
colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))

for (name, group), color in zip(groups, colors):
    ax.scatter(group['Toluene_frac'], group['N-Heptane_frac'], marker='o', color=color, label=name)
    ax.plot(group['Toluene_frac'], group['N-Heptane_frac'], linestyle='--', color=color)

ax.set_tlabel("Toluene")
ax.set_llabel("N-Heptane")
ax.set_rlabel("IL")
ax.clear_ticks()
ax.legend(title='Group No.', loc='upper right')

# Save the equilateral triangle plot
plt.savefig('novice.png')

# Create the right-angled triangle plot
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_title("Liquid-Liquid Phase Diagram", fontsize=16)

for (name, group), color in zip(groups, colors):
    ax2.scatter(group['Toluene_frac'], group['IL_frac'], marker='o', color=color, label=name)
    ax2.plot(group['Toluene_frac'], group['IL_frac'], linestyle='--', color=color)

ax2.set_xlabel("Toluene")
ax2.set_ylabel("IL")
ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(title='Group No.', loc='upper right')

# Save the right-angled triangle plot
plt.savefig('novice_right_angle.png')

# Show both plots
plt.tight_layout()
plt.show()