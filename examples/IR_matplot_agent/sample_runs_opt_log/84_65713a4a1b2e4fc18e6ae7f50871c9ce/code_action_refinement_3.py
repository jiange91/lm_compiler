import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from CSV
data = pd.read_csv('data.csv')

# Print the columns to check their names
print("Columns in the CSV file:", data.columns)

# Assuming the correct column names are 'Temp1', 'Pressure1', 'Temp2', 'Pressure2'
try:
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set axis labels
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (Pa)')

    # Plot Phase Boundaries
    ax.plot(data['Temp1'], data['Pressure1'], label='Phase Boundary (Solid-Liquid-Gas)', color='blue')
    ax.plot(data['Temp2'], data['Pressure2'], label='Phase Boundary (Solid-Liquid)', color='orange')

    # Mark Special Points
    ax.plot(273.16, 611.657, 'ro', label='Triple Point (273.16 K, 611.657 Pa)')
    ax.plot(647.396, 22.064e6, 'ro', label='Critical Point (647.396 K, 22.064 MPa)')

    # Draw Freezing and Boiling Points
    ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
    ax.axvline(x=100, color='orange', linestyle='--', label='Boiling Point (100°C)')

    # Color Different Phases
    ax.fill_betweenx(y=[0, 611.657], x1=-50, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
    ax.fill_betweenx(y=[611.657, 22.064e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
    ax.fill_betweenx(y=[22.064e6, 10e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

    # Set Logarithmic Scale for Pressure
    ax.set_yscale('log')

    # Add Grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add Legend and Title
    ax.set_title('Phase Diagram of Water')
    ax.legend()

    # Save the Plot
    plt.savefig('novice_final.png')
    plt.show()

except KeyError as e:
    print(f"KeyError: {e}. Please check the column names in your CSV file.")