import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('data.csv')

# Calculate the weighted average life expectancy
weighted_avg_lifeExp = (data['lifeExp'] * data['pop']).sum() / data['pop'].sum()

# Create the sunburst plot
fig = px.sunburst(data, 
                  path=['continent', 'country'], 
                  values='pop', 
                  color='lifeExp', 
                  color_continuous_scale=px.colors.sequential.RdBu,
                  color_continuous_midpoint=weighted_avg_lifeExp)

# Update layout and title
fig.update_layout(title='Sunburst Plot of Countries by Continent')

# Save the plot as a PNG file
fig.write_image("novice_final.png")