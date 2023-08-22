# import networkx as nx
# import matplotlib.pyplot as plt

# # Create the graph
# G = nx.Graph()

# # Add variable nodes
# variables = ['x', 'y', 'z']
# for var in variables:
#     G.add_node(var, color='lightblue')

# # Add factor nodes
# factors = ['f1', 'f2']
# for factor in factors:
#     G.add_node(factor, color='lightcoral')

# # Add edges between factors and their corresponding variables
# G.add_edges_from([('f1', 'x'), ('f1', 'y'), ('f2', 'y'), ('f2', 'z')])

# # Get node colors from the graph node attributes
# colors = [G.nodes[node]['color'] for node in G.nodes()]

# # Draw the graph
# pos = nx.spring_layout(G)
# fig, ax = plt.subplots()
# for node, (x, y) in pos.items():
#     ax.scatter(x, y, c=G.nodes[node]['color'], s=2000, label=node)
#     ax.text(x, y, node, horizontalalignment='center', verticalalignment='center')

# for edge in G.edges():
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     ax.plot([x0, x1], [y0, y1], c='k')

# plt.title("Factor Graph")
# plt.show()

import networkx as nx
import matplotlib.pyplot as plt

# Create the graph
G = nx.Graph()

# Add variable nodes
variables = ['x', 'y', 'z']
G.add_nodes_from(variables, color='lightblue')

# Add factor nodes
factors = ['f1', 'f2']
G.add_nodes_from(factors, color='lightcoral')

# Add edges between factors and their corresponding variables
G.add_edges_from([('f1', 'x'), ('f1', 'y'), ('f2', 'y'), ('f2', 'z')])

# Draw the graph
colors = ['lightblue' if n in variables else 'lightcoral' for n in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000)
plt.title("Factor Graph")
plt.show()
