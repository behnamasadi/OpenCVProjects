A **Factor Graph** is a bipartite graph that represents the factorization of a function. It is used in many fields, particularly for inference in statistical models, and is especially popular in the domain of graphical models and machine learning. In a factor graph, there are two types of nodes:

1. **Variable Nodes**: Represent variables in your model.
2. **Factor Nodes**: Represent factors or functions that operate on one or more of these variables.

Edges in the graph connect factor nodes to the variable nodes they involve. A factor graph visually and structurally captures how the global function decomposes into a product of local functions.

To explain with a simple example: consider a function \( f(x, y, z) \) that can be factorized as:
\[ f(x, y, z) = f_1(x, y) \times f_2(y, z) \]

Here:
- \( x, y, z \) are the variables.
- \( f_1 \) is a factor involving variables \( x \) and \( y \).
- \( f_2 \) is a factor involving variables \( y \) and \( z \).

The factor graph will have three variable nodes (for \( x, y, z \)), and two factor nodes (for \( f_1, f_2 \)).

Let's visualize this factor graph using Python:

```
conda install -c anaconda networkx
pip install --upgrade networkx matplotlib
```



``` python
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
```

In the visualized graph, you'll see:
- Blue nodes represent variables (x, y, z).
- Red nodes represent factors (f1, f2).
- Edges connect factors to the variables they involve.

This is a very simple example, and real-world factor graphs can be much more complex. Factor graphs are particularly useful in belief propagation and other inference algorithms, where the structure of the graph helps to systematically update beliefs about the variables based on observed data and the relationships encoded by the factors.





## Message passing algorithm

## Computing Marginal

## Markov Network

## Bayes Network


## Belief propagation


## Factor graph vs pose graph
A factor graph can be seen as a generalization of a pose graph when considered as a representation for graph based slam. In a factor graph, we can model more than with pose graphs a such as factors connected to a single node, for example useful for gps/gas’s measurements.


- Factor Graph - 5 Minutes with Cyrill
Refs: [1](https://www.youtube.com/watch?v=uuiaqGLFYa4&t=145s)


