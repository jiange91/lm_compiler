import json
import plotly.graph_objects as go

# Function to convert hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove the '#' character
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Convert to RGB tuple

# Load the JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract the relevant parts of the data
energy_data = data['data'][0]  # Access the first (and only) data object
layout_data = data['layout']    # Access layout settings

# Extract nodes and links
nodes = energy_data['node']  # Assuming 'node' contains node information
links = energy_data['link']   # Assuming 'link' contains link information

# Prepare node labels and colors
node_labels = nodes['label']  # List of node labels
node_colors = nodes['color']   # List of node colors

# Prepare link properties
link_source = links['source']   # List of source indices for links
link_target = links['target']    # List of target indices for links
link_values = links['value']     # List of values for links
link_colors = links['color']     # List of colors for links

# Set transparency for links
link_opacity = 0.5  # Adjust this value for desired transparency

# Create a lighter color for each link based on the source color
link_colors_rgba = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {link_opacity})' 
                    for c in [hex_to_rgb(color) for color in link_colors]]

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
        value=link_values,    # Values of links
        color=link_colors_rgba  # Link colors with transparency
    )
))

# Update layout with title and other settings
fig.update_layout(
    title_text=layout_data.get('title', 'Energy Flow Diagram'),  # Title from layout data
    font=dict(size=10),  # Font size
    height=600,  # Height of the diagram
    width=800,  # Width of the diagram
)

# Save the figure to a PNG file
fig.write_image("novice.png")

# Show the figure
fig.show()