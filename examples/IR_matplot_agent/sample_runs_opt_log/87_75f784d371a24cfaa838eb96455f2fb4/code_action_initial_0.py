import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv('data.csv')

# Check if the required columns are present
required_columns = ['country', 'continent', 'lifeExp', 'pop']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing one or more required columns in the data.")

# Calculate the weighted average lifespan
data['weighted_lifeExp'] = data['lifeExp'] * data['pop']
total_population = data['pop'].sum()
average_lifeExp = data['weighted_lifeExp'].sum() / total_population

# Create the sunburst plot
fig = px.sunburst(
    data_frame=data,
    path=['continent', 'country'],  # Hierarchical structure
    values='pop',                    # Size by population
    color='lifeExp',                 # Color by life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Color scale from red to blue
    color_continuous_midpoint=average_lifeExp,  # Set midpoint to average life expectancy
    title='Sunburst Plot of Countries by Continent'
)

# Update layout to include a legend
fig.update_layout(
    coloraxis_colorbar=dict(
        title='Life Expectancy',
        tickvals=[data['lifeExp'].min(), average_lifeExp, data['lifeExp'].max()],
        ticktext=[f"{data['lifeExp'].min():.1f}", f"{average_lifeExp:.1f}", f"{data['lifeExp'].max():.1f}"]
    )
)

# Save the plot to a file
fig.write_image("novice.png")

# Show the plot
fig.show()