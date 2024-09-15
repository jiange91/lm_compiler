import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Import the data
data = pd.read_csv('data.csv')

# Step 2: Extract the necessary columns
temperature1 = data.iloc[:, 0]  # Column 1 for temperature
pressure1 = data.iloc[:, 1]     # Column 2 for pressure
temperature2 = data.iloc[:, 2]  # Column 3 for temperature
pressure2 = data.iloc[:, 3]     # Column 4 for pressure

# Step 3: Set up the plot
fig, ax = plt.subplots()

# Step 4: Plot the phase boundaries
ax.plot(temperature1, pressure1, label='Solid-Liquid-Gas Boundary')
ax.plot(temperature2, pressure2, label='Solid-Liquid Boundary')

# Step 5: Mark special points
triple_point = (273.16, 611.657)  # Triple point in K and Pa
critical_point = (647.396, 22.064e6)  # Critical point in K and Pa

ax.plot(*triple_point, 'go', label='Triple Point')
ax.plot(*critical_point, 'ro', label='Critical Point')

# Step 6: Draw vertical lines for freezing and boiling points
ax.axvline(x=273.15, color='r', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=373.15, color='r', linestyle='--', label='Boiling Point (100°C)')

# Step 7: Label and color the regions
ax.fill_between(temperature1, pressure1, pressure2, where=(pressure1 > pressure2), color='blue', alpha=0.3, label='Solid')
ax.fill_between(temperature1, pressure1, pressure2, where=(pressure1 < pressure2), color='green', alpha=0.3, label='Liquid')
ax.fill_between(temperature1, pressure1, 1e8, where=(pressure1 < 1e8), color='yellow', alpha=0.3, label='Gas')

# Step 8: Add grid and logarithmic scale
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Step 9: Set labels and title
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')
ax.legend()

# Step 10: Save the plot
plt.savefig('plot.png')

# Show the plot
plt.show()