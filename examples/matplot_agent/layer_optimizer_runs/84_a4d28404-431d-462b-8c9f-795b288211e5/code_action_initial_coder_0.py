import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Import the data
data = pd.read_csv('data.csv')

# Step 2: Extract the necessary columns
temp_slg = data.iloc[:, 0]  # Temperature for solid-liquid-gas boundary (Celsius)
pressure_slg = data.iloc[:, 1]  # Pressure for solid-liquid-gas boundary (Pascals)
temp_sl = data.iloc[:, 2]  # Temperature for solid-liquid boundary (Celsius)
pressure_sl = data.iloc[:, 3]  # Pressure for solid-liquid boundary (Pascals)

# Step 3: Set up the plot
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Phase Diagram of Water')

# Step 4: Plot the phase boundaries
ax.plot(temp_slg + 273.15, pressure_slg, label='Solid-Liquid-Gas Boundary')
ax.plot(temp_sl + 273.15, pressure_sl, label='Solid-Liquid Boundary')

# Step 5: Mark special points
triple_point = (273.16, 611.657)  # Triple point (K, Pa)
critical_point = (647.396, 22.064e6)  # Critical point (K, Pa)
ax.plot(*triple_point, 'go', label='Triple Point (273.16 K, 611.657 Pa)')
ax.plot(*critical_point, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

# Step 6: Draw vertical lines for freezing and boiling points
ax.axvline(x=273.15, color='red', linestyle='--', label='Freezing Point (0°C)')
ax.axvline(x=373.15, color='red', linestyle='--', label='Boiling Point (100°C)')

# Step 7: Label and color the regions
ax.fill_betweenx(pressure_slg, temp_slg + 273.15, 273.15, where=(temp_slg + 273.15 <= 273.15), color='blue', alpha=0.3, label='Solid')
ax.fill_betweenx(pressure_slg, 273.15, 373.15, where=(temp_slg + 273.15 >= 273.15) & (temp_slg + 273.15 <= 373.15), color='green', alpha=0.3, label='Liquid')
ax.fill_betweenx(pressure_slg, 373.15, temp_slg + 273.15, where=(temp_slg + 273.15 >= 373.15), color='orange', alpha=0.3, label='Gas')

# Step 8: Add grid and finalize the plot
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()

# Step 9: Save the plot
plt.savefig('plot.png')

# Show the plot
plt.show()