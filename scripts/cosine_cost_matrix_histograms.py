import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

# Define the histograms
A = np.array([1, 2, 3])
B = np.array([2, 3, 4])
C = np.array([4, 5, 6])

# Function to calculate cosine similarity
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Function to calculate cosine distance
def cosine_distance(x, y):
    return 1 - cosine_similarity(x, y)

# Calculate the cost matrix
histograms = [A, B, C]
n = len(histograms)
cost_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        cost_matrix[i, j] = cosine_distance(histograms[i], histograms[j])

cost_matrix



# Set the labels for the histograms
labels = ['A', 'B', 'C']

# Plotting the cost matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cost_matrix, annot=True, fmt=".4f", xticklabels=labels, yticklabels=labels, cmap="coolwarm")
plt.title('Cosine Distance Cost Matrix')
plt.xlabel('Histograms')
plt.ylabel('Histograms')
plt.show()
