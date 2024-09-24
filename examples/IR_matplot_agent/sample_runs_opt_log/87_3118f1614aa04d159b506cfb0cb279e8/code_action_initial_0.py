import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('data.csv')

# Group by continent and country, summing the population
grouped_data = data.groupby(['continent', 'country']).agg({
    'pop': 'sum',
    'lifeExp': 'mean'
}).reset_index()

# Calculate the weighted average lifespan
weighted_avg_lifeExp = (grouped_data['lifeExp'] * grouped_data['pop']).sum() / grouped_data['pop'].sum()
print(f'Weighted Average Life Expectancy: {weighted_avg_lifeExp}')

# Create the sunburst plot
fig = px.sunburst(
    grouped_data,
    path=['continent', 'country'],  # Hierarchical structure
    values='pop',                    # Size of segments based on population
    color='lifeExp',                 # Color segments based on life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Color scale from red to blue
    color_continuous_midpoint=weighted_avg_lifeExp,    # Set midpoint to the weighted average
    title='Sunburst Plot of Life Expectancy by Country and Continent'
)

# Update layout to include a legend
fig.update_layout(coloraxis_colorbar=dict(title='Life Expectancy'))

# Save the plot to a PNG file
fig.write_image("novice.png")

# Show the plot
fig.show()