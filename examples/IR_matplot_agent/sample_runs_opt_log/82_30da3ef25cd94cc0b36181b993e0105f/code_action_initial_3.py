import json
import plotly.graph_objects as go

# Load the JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract the relevant parts of the data
energy_data = data['data'][0]  # Access the first (and only) data object
layout = data['layout']  # Access layout settings

# Extract nodes and links
nodes = energy_data['node']  # Ensure nodes is a list
links = energy_data['link']  # Ensure links is a list

# Prepare node labels and colors
node_labels = [node['label'] for node in nodes]  # Assuming each node has a 'label' key
node_colors = [node.get('color', 'rgba(0,0,0,0.5)') for node in nodes]  # Default color if not specified

# Prepare link source, target, values, and colors
link_source = [link['source'] for link in links]  # Assuming 'source' key exists
link_target = [link['target'] for link in links]  # Assuming 'target' key exists
link_values = [link['value'] for link in links]  # Assuming 'value' key exists
link_colors = [link.get('color', 'rgba(0,0,0,0.5)') for link in links]  # Default color if not specified

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,  # Padding between nodes
        thickness=20,  # Thickness of nodes
        line=dict(color='black', width=0.5),  # Node border color and width
        label=node_labels,  # Node labels
        color=node_colors  # Node colors
    ),
    link=dict(
        source=link_source,  # Indices of source nodes
        target=link_target,  # Indices of target nodes
        value=link_values,  # Values of links
        color=link_colors  # Link colors
    )
))

# Update layout settings
fig.update_layout(
    title=layout.get('title', 'Energy Flow Diagram'),  # Title from layout or default
    font=dict(size=10),  # Font size
    height=layout.get('height', 600),  # Height from layout or default
    width=layout.get('width', 800)  # Width from layout or default
)

# Save the figure to a PNG file
fig.write_image("novice.png")

# Show the figure
fig.show()