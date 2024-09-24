import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('data.csv')

# Create a new DataFrame for the sunburst plot
sunburst_data = data[['continent', 'country', 'lifeExp', 'pop']]

# Calculate the weighted average life expectancy
weighted_avg_lifeExp = (sunburst_data['lifeExp'] * sunburst_data['pop']).sum() / sunburst_data['pop'].sum()

# Create the sunburst plot
fig = px.sunburst(
    sunburst_data,
    path=['continent', 'country'],
    values='pop',
    color='lifeExp',
    color_continuous_scale=px.colors.sequential.RdBu,
    color_continuous_midpoint=weighted_avg_lifeExp,
    title='Sunburst Plot of Life Expectancy by Country and Continent'
)

# Update layout to include a legend
fig.update_layout(
    coloraxis_colorbar=dict(
        title='Life Expectancy',
        tickvals=[40, 50, 60, 70, 80, 90],
        ticktext=['40', '50', '60', '70', '80', '90']
    )
)

# Save the plot to a PNG file
fig.write_image("novice_final.png")