import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Define the histograms
# Reshape for sklearn function compatibility
histogram_A = np.array([10, 20, 15]).reshape(1, -1)
histogram_B = np.array([10, 15, 20]).reshape(1, -1)
histogram_C = np.array([5, 25, 10]).reshape(1, -1)

# Initialize an empty similarity matrix
num_histograms = 3
similarity_matrix = np.zeros((num_histograms, num_histograms))

# Populate the similarity matrix using cosine similarity
histograms = [histogram_A, histogram_B, histogram_C]
for i in range(num_histograms):
    for j in range(num_histograms):
        # Compute cosine similarity and subtract from 1 to represent cost
        similarity_matrix[i, j] = cosine_similarity(
            histograms[i], histograms[j])

# Convert similarity to cost
cost_matrix = 1 - similarity_matrix

# Plot the cost matrix
fig, ax = plt.subplots()
cax = ax.matshow(cost_matrix, cmap='viridis')

# Add colorbar to explain the color encoding
fig.colorbar(cax)

# Define histogram names for labeling
histogram_names = ['A', 'B', 'C']

# Set axis labels with correct labeling
ax.set_xticks(np.arange(len(histogram_names)))
ax.set_yticks(np.arange(len(histogram_names)))
ax.set_xticklabels(histogram_names)
ax.set_yticklabels(histogram_names)

# Title and labels
plt.title('Cosine Cost Matrix')
plt.xlabel('Histograms')
plt.ylabel('Histograms')

# Show plot
plt.show()
