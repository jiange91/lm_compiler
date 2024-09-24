import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Print the columns to check their names
print("Columns in the CSV file:", data.columns)

# Assuming the correct column names are 'Temp1', 'Pressure1', 'Temp2', 'Pressure2'
# Adjust these names based on the actual output from the print statement above
try:
    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_title('Phase Diagram of Water')

    # Convert Pressure to Logarithmic Scale
    ax.set_yscale('log')

    # Plot the Phase Boundaries
    ax.plot(data['Temp1'], data['Pressure1'], label='Gas-Liquid Boundary', color='blue')  # Adjust column names
    ax.plot(data['Temp2'], data['Pressure2'], label='Solid-Liquid Boundary', color='green')  # Adjust column names

    # Mark the Special Points
    ax.plot(0, 611.657, 'ro', label='Triple Point')
    ax.plot(647.396 - 273.15, 22.064 * 1e6, 'ro', label='Critical Point')  # Convert MPa to Pa

    # Draw Freezing and Boiling Points
    ax.axvline(x=0, color='red', linestyle='--', label='Freezing Point (0°C)')
    ax.axvline(x=100, color='red', linestyle='--', label='Boiling Point (100°C)')

    # Color the Phases
    ax.fill_betweenx(y=[0, 611.657], x1=-40, x2=0, color='lightblue', alpha=0.5, label='Solid Phase')
    ax.fill_betweenx(y=[611.657, 22.064 * 1e6], x1=0, x2=100, color='lightgreen', alpha=0.5, label='Liquid Phase')
    ax.fill_betweenx(y=[22.064 * 1e6, 1e7], x1=100, x2=200, color='lightyellow', alpha=0.5, label='Gas Phase')

    # Add a Grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a Legend
    ax.legend()

    # Save the Plot
    plt.savefig('novice_final.png')
    plt.show()

except KeyError as e:
    print(f"KeyError: {e}. Please check the column names in your CSV file.")