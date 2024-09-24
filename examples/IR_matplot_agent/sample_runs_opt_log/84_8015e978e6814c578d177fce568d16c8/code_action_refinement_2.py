import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Data
data = pd.read_csv('data.csv')

# Check the column names
print(data.columns)

# Set Up the Plot
fig, ax = plt.subplots()

# Define Axes
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Pa)')

# Logarithmic Scale for Pressure
ax.set_yscale('log')

# Plot Phase Boundaries
ax.plot(data['Temperature'], data['Pressure'], label='Phase Boundary', color='blue')
ax.plot(data['Solid-Liquid_Temperature'], data['Solid-Liquid_Pressure'], label='Solid-Liquid Boundary', color='green')
ax.plot(data['Solid-Gas_Temperature'], data['Solid-Gas_Pressure'], label='Solid-Gas Boundary', color='orange')

# Mark Special Points
ax.plot(273.16, 611.657, 'ro', label='Triple Point')
ax.plot(647.396, 22.064e6, 'ro', label='Critical Point')

# Draw Freezing and Boiling Points
ax.axvline(x=273.15, color='red', linestyle='--', label='Freezing Point')
ax.axvline(x=373.15, color='red', linestyle='--', label='Boiling Point')

# Color the Phases
ax.fill_betweenx(y=[1e3, 1e7], x1=0, x2=273.15, color='lightblue', alpha=0.5, label='Solid Phase')
ax.fill_betweenx(y=[1e3, 1e7], x1=273.15, x2=373.15, color='lightgreen', alpha=0.5, label='Liquid Phase')
ax.fill_betweenx(y=[1e3, 1e7], x1=373.15, x2=647.396, color='lightyellow', alpha=0.5, label='Gas Phase')

# Add a Grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a Legend
ax.legend()

# Set Title
ax.set_title('Phase Diagram of Water')

# Save the Plot
plt.savefig('novice_final.png')

# Show the Plot
plt.show()