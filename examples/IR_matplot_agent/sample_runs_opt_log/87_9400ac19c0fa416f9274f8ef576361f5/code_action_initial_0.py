import pandas as pd
import plotly.express as px

# Step 1: Load the data from CSV
data = pd.read_csv('data.csv')

# Step 2: Validate required columns
required_columns = ['country', 'continent', 'lifeExp', 'pop']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing one or more required columns in the data.")

# Step 3: Group by continent and country, summing population and averaging life expectancy
grouped_data = data.groupby(['continent', 'country']).agg({
    'pop': 'sum',        # Total population for each country
    'lifeExp': 'mean'    # Average life expectancy for each country
}).reset_index()

# Step 4: Calculate the weighted average lifespan
weighted_avg_lifeExp = (grouped_data['lifeExp'] * grouped_data['pop']).sum() / grouped_data['pop'].sum()
print(f'Weighted Average Life Expectancy: {weighted_avg_lifeExp}')

# Step 5: Create the sunburst plot
fig = px.sunburst(
    grouped_data,
    path=['continent', 'country'],  # Hierarchical structure
    values='pop',                    # Size of segments based on population
    color='lifeExp',                 # Color segments based on life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Color scale from red to blue
    color_continuous_midpoint=weighted_avg_lifeExp,    # Set midpoint to the weighted average
    title='Sunburst Plot of Life Expectancy by Country and Continent'
)

# Step 6: Update layout to include a legend
fig.update_layout(coloraxis_colorbar=dict(title='Life Expectancy'))

# Step 7: Save the plot to a PNG file
fig.write_image("novice.png")

# Step 8: Show the plot
fig.show()