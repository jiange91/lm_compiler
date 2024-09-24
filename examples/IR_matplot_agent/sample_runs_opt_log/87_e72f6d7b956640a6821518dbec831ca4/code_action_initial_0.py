# Step 1: Install Required Libraries
# This step is done via pip in the terminal, not in the code.

# Step 2: Import Libraries
import pandas as pd
import plotly.express as px

# Step 3: Load the Data
data = pd.read_csv('data.csv')

# Step 4: Data Preparation
# Display the first few rows of the DataFrame to understand its structure
print(data.head())

# Calculate the weighted average lifespan
data['weighted_lifeExp'] = data['lifeExp'] * data['pop']  # Calculate weighted life expectancy
total_population = data['pop'].sum()  # Total population
average_lifeExp = data['weighted_lifeExp'].sum() / total_population  # Weighted average life expectancy

print(f"Average Life Expectancy (weighted): {average_lifeExp:.2f}")

# Step 5: Create the Sunburst Plot
fig = px.sunburst(
    data_frame=data,
    path=['continent', 'country'],  # Hierarchical structure
    values='pop',                    # Size of segments based on population
    color='lifeExp',                 # Color segments based on life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Color scale from red to blue
    color_continuous_midpoint=average_lifeExp,  # Set midpoint to average life expectancy
    title='Sunburst Plot of Life Expectancy and Population by Country'
)

# Step 6: Add Legend and Save the Plot
# Update layout to include a legend
fig.update_layout(
    coloraxis_colorbar=dict(
        title='Life Expectancy',
        tickvals=[data['lifeExp'].min(), average_lifeExp, data['lifeExp'].max()],
        ticktext=[f"{data['lifeExp'].min():.1f}", f"{average_lifeExp:.1f}", f"{data['lifeExp'].max():.1f}"]
    )
)

# Save the plot as a PNG file
fig.write_image("novice.png")

# Step 7: Display the Plot
fig.show()