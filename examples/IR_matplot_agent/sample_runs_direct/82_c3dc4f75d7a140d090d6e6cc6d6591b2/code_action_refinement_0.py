import json
import plotly.graph_objects as go

# Step 1: Open the Data File
with open('data.json') as f:
    data = json.load(f)

# Step 2: Extract Data for Plotting
energy_data = data['data'][0]  # Assuming there's only one primary data object
layout = data['layout']

# Step 3: Color Settings
link_color = 'rgba(0, 0, 255, 0.5)'  # Example color with transparency
node_color = 'rgba(255, 0, 0, 0.5)'  # Example color for nodes

# Step 4: Set Up Nodes
nodes = energy_data['node']
for node in nodes:
    node['color'] = node_color  # Assign color to each node
    node['thickness'] = 20  # Adjust thickness as needed

# Step 5: Set Up Links
links = energy_data['link']
for link in links:
    link['color'] = link_color  # Assign color to each link
    link['value'] = link['value']  # Ensure value is correctly set

# Step 6: Create the Sankey Diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=[node['label'] for node in nodes],
        color=[node['color'] for node in nodes]
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links],
        color=[link['color'] for link in links]
    )
))

fig.update_layout(title_text="Energy Flow Diagram", font_size=10)

# Step 7: Save the Plot
fig.write_image("novice_final.png")