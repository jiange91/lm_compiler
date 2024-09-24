import pandas as pd
import plotly.express as px

# Load the data from CSV
data = pd.read_csv('data.csv')

# Check for missing values and drop them
required_columns = ['country', 'continent', 'lifeExp', 'pop']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing one or more required columns in the data.")
data = data[required_columns].dropna()

# Calculate the weighted average life expectancy
weighted_avg_lifeExp = (data['lifeExp'] * data['pop']).sum() / data['pop'].sum()
print(f'Weighted Average Life Expectancy: {weighted_avg_lifeExp}')

# Create the sunburst plot
fig = px.sunburst(
    data_frame=data,
    path=['continent', 'country'],  # Hierarchical structure
    values='pop',                    # Size of segments based on population
    color='lifeExp',                 # Color segments based on life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Color scale from red to blue
    color_continuous_midpoint=weighted_avg_lifeExp,  # Set midpoint to weighted average
    title='Sunburst Plot of Life Expectancy by Country and Continent'
)

# Update layout to include a legend
fig.update_layout(
    coloraxis_colorbar=dict(
        title='Life Expectancy',
        tickvals=[data['lifeExp'].min(), weighted_avg_lifeExp, data['lifeExp'].max()],
        ticktext=[f"{data['lifeExp'].min():.1f}", f"{weighted_avg_lifeExp:.1f}", f"{data['lifeExp'].max():.1f}"]
    )
)

# Save the plot to a PNG file
fig.write_image("novice.png")

# Show the plot
fig.show()