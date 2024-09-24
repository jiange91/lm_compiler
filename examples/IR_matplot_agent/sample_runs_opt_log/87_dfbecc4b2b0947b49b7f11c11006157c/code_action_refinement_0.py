import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('data.csv')

# Select relevant columns
data = data[['country', 'continent', 'lifeExp', 'pop']]

# Calculate the weighted average life expectancy
weighted_avg_lifeExp = (data['lifeExp'] * data['pop']).sum() / data['pop'].sum()

# Create the sunburst plot
fig = px.sunburst(data_frame=data,
                  path=['continent', 'country'],
                  values='pop',
                  color='lifeExp',
                  color_continuous_scale=px.colors.sequential.RdBu,
                  color_continuous_midpoint=weighted_avg_lifeExp)

# Update layout for title
fig.update_layout(title_text='Sunburst Plot of Life Expectancy by Country and Continent')

# Add a color bar for life expectancy
fig.update_traces(hoverinfo='label+value+percent entry', textinfo='label+value')

# Save the plot to a PNG file
fig.write_image("novice_final.png")