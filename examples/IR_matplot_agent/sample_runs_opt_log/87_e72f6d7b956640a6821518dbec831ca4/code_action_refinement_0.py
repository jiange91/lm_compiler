import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('data.csv')

# Calculate average life expectancy weighted by population
avg_life_exp = (data['lifeExp'] * data['pop']).sum() / data['pop'].sum()

# Create the sunburst plot
fig = px.sunburst(data,
                  path=['continent', 'country'],
                  values='pop',
                  color='lifeExp',
                  color_continuous_scale=px.colors.sequential.RdBu,
                  color_continuous_midpoint=avg_life_exp,
                  title='Sunburst Plot of Life Expectancy and Population by Country')

# Add a legend
fig.update_layout(coloraxis_colorbar=dict(title='Life Expectancy'))

# Save the plot
fig.write_image("novice_final.png")