# Step 1: Install Required Libraries
# Make sure to run this in your terminal or command prompt
# pip install pandas plotly

# Step 2: Import Libraries
import pandas as pd
import plotly.express as px

# Step 3: Load the Data
data = pd.read_csv('data.csv')

# Step 4: Data Preparation
# Check if the required columns are present
required_columns = ['country', 'continent', 'lifeExp', 'pop']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Missing one or more required columns in the data.")

# Calculate the weighted average lifespan
data['weighted_lifeExp'] = data['lifeExp'] * data['pop']
total_population = data['pop'].sum()
average_lifeExp = data['weighted_lifeExp'].sum() / total_population

# Step 5: Create the Sunburst Plot
fig = px.sunburst(
    data_frame=data,
    path=['continent', 'country'],  # Hierarchical structure
    values='pop',                    # Size by population
    color='lifeExp',                 # Color by life expectancy
    color_continuous_scale=px.colors.sequential.RdBu,  # Color scale from red to blue
    color_continuous_midpoint=average_lifeExp,  # Set midpoint to average life expectancy
    title='Sunburst Plot of Countries by Continent'
)

# Step 6: Add Legend and Update Layout
# Update layout to include a legend
fig.update_layout(
    coloraxis_colorbar=dict(
        title='Life Expectancy',
        tickvals=[data['lifeExp'].min(), average_lifeExp, data['lifeExp'].max()],
        ticktext=[f"{data['lifeExp'].min():.1f}", f"{average_lifeExp:.1f}", f"{data['lifeExp'].max():.1f}"]
    )
)

# Step 7: Save the Plot as a PNG File
plot_file_name = 'novice.png'
fig.write_image(plot_file_name)

# Step 8: Show the Plot
fig.show()