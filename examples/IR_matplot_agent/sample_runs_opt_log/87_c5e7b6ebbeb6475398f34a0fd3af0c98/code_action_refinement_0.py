import pandas as pd
import plotly.express as px

# Step 1: Load the Data
data = pd.read_csv('data.csv')

# Step 2: Check Data Columns
print(data.columns)

# Step 3: Data Aggregation
continent_data = data.groupby('continent').agg({'pop': 'sum', 'lifeExp': 'mean'}).reset_index()
country_data = data.groupby(['continent', 'country']).agg({'pop': 'sum', 'lifeExp': 'mean'}).reset_index()

# Step 4: Create the Sunburst Plot
fig = px.sunburst(
    country_data,
    path=['continent', 'country'],
    values='pop',
    color='lifeExp',
    color_continuous_scale=px.colors.sequential.RdBu,
    color_continuous_midpoint=country_data['lifeExp'].mean()
)

# Step 5: Customize the Plot
fig.update_layout(title='Sunburst Plot of Countries by Continent')
fig.update_traces(textinfo='label+percent entry')

# Step 6: Save the Plot
fig.write_image("novice_final.png")