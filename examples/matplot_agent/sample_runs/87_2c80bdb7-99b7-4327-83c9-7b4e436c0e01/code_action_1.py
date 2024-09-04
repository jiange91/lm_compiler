import pandas as pd
import plotly.express as px

# Load the dataset

df = pd.read_csv('data.csv')
df.dropna(subset=['country', 'continent', 'lifeExp', 'pop'], inplace=True)

# Calculate the average life expectancy weighted by population
avg_life_exp = (df['lifeExp'] * df['pop']).sum() / df['pop'].sum()

# Create the sunburst plot
fig = px.sunburst(
    df,
    path=['continent', 'country'],  # Hierarchical organization
    values='pop',  # Size segments based on population
    color='lifeExp',  # Color scale based on life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Red to blue color scale
    color_continuous_midpoint=avg_life_exp,  # Central value of the color scale
    title='Sunburst Plot of Countries by Continent'
)

# Update layout for better visibility
fig.update_layout(
    legend_title_text='Life Expectancy',  # Legend for life expectancy
    title_x=0.5  # Center the title
)

# Save the plot to a PNG file
fig.write_image('plot.png')

# Show the plot
fig.show()