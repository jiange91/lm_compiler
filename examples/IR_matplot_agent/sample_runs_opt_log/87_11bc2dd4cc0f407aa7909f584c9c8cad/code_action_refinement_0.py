import pandas as pd
import plotly.express as px

# Step 1: Load the Data
data = pd.read_csv('data.csv')

# Step 2: Check Data Columns
print(data.columns)

# Step 3: Create Hierarchical Data
data['size'] = data['pop']  # Use population for segment size

# Step 4: Calculate Average Life Expectancy
avg_life_exp = (data['lifeExp'] * data['pop']).sum() / data['pop'].sum()

# Step 5: Create the Sunburst Plot
fig = px.sunburst(data_frame=data,
                  path=['continent', 'country'],
                  values='size',
                  color='lifeExp',
                  color_continuous_scale=px.colors.sequential.RdBu,
                  color_continuous_midpoint=avg_life_exp)

# Step 6: Set Title and Legend
fig.update_layout(title='Sunburst Plot of Countries by Continent',
                  coloraxis_colorbar=dict(title='Life Expectancy'))

# Step 7: Save the Plot
fig.write_image("novice_final.png")