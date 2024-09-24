import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('data.csv')

# Check the columns
print(data.columns)

# Group data by continent and country
data_grouped = data.groupby(['continent', 'country']).agg({'lifeExp': 'mean', 'pop': 'sum'}).reset_index()

# Calculate the weighted average lifespan
weighted_avg_lifeExp = (data_grouped['lifeExp'] * data_grouped['pop']).sum() / data_grouped['pop'].sum()

# Create the sunburst plot
fig = px.sunburst(
    data_grouped,
    path=['continent', 'country'],
    values='pop',
    color='lifeExp',
    color_continuous_scale=px.colors.sequential.RdBu,
    color_continuous_midpoint=weighted_avg_lifeExp,
    title='Sunburst Plot of Life Expectancy by Country and Continent'
)

# Update layout to include a legend
fig.update_layout(coloraxis_colorbar=dict(title='Life Expectancy'))

# Save the plot
fig.write_image("novice_final.png")