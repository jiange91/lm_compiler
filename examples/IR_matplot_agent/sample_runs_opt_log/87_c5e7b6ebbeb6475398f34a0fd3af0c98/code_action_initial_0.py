import pandas as pd
import plotly.express as px

# Step 1: Load the data
data = pd.read_csv('data.csv')

# Step 2: Check if the required columns are present
required_columns = ['country', 'continent', 'lifeExp', 'pop']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing one or more required columns in the data.")

# Step 3: Calculate the weighted average lifespan
data['weighted_lifeExp'] = data['lifeExp'] * data['pop']
total_population = data['pop'].sum()
average_lifeExp = data['weighted_lifeExp'].sum() / total_population

# Step 4: Create the sunburst plot
fig = px.sunburst(
    data_frame=data,
    path=['continent', 'country'],  # Hierarchical structure
    values='pop',                    # Size by population
    color='lifeExp',                 # Color by life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Color scale from red to blue
    color_continuous_midpoint=average_lifeExp,  # Set midpoint to average life expectancy
    title='Sunburst Plot of Countries by Continent'
)

# Step 5: Update layout to include a legend
fig.update_layout(
    coloraxis_colorbar=dict(
        title='Life Expectancy',
        tickvals=[data['lifeExp'].min(), average_lifeExp, data['lifeExp'].max()],
        ticktext=[f"{data['lifeExp'].min():.1f}", f"{average_lifeExp:.1f}", f"{data['lifeExp'].max():.1f}"]
    )
)

# Step 6: Save the plot to a PNG file
fig.write_image("novice.png")

# Step 7: Show the plot
fig.show()