import pandas as pd
import plotly.express as px

# Step 1: Load the data
data = pd.read_csv('data.csv')

# Step 2: Prepare the data
grouped_data = data.groupby(['continent', 'country']).agg({'lifeExp': 'mean', 'pop': 'sum'}).reset_index()

# Step 3: Calculate the average life expectancy
weighted_avg_lifeExp = (grouped_data['lifeExp'] * grouped_data['pop']).sum() / grouped_data['pop'].sum()

# Step 4: Create the sunburst plot
fig = px.sunburst(
    grouped_data,
    path=['continent', 'country'],
    values='pop',
    color='lifeExp',
    color_continuous_scale=px.colors.sequential.RdBu,
    color_continuous_midpoint=weighted_avg_lifeExp,
    title='Sunburst Plot of Countries by Continent'
)

# Step 5: Add a legend
fig.update_layout(coloraxis_colorbar=dict(title='Life Expectancy'))

# Step 6: Save the plot
fig.write_image("novice_final.png")