import numpy as np
import matplotlib.pyplot as plt

# Define a small corpus and a document of interest
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox",
    "the lazy dog",
    "the fox"
]
document_of_interest = "the quick brown fox jumps"

# Calculate TF-IDF
# First, let's calculate term frequency (TF) for the document of interest

# Split the document into terms
terms_in_document = document_of_interest.split()

# Calculate the total number of terms in the document
total_terms_in_document = len(terms_in_document)

# Calculate the frequency of each term in the document
tf = {term: terms_in_document.count(term) / total_terms_in_document for term in terms_in_document}

# Now, let's calculate inverse document frequency (IDF) for terms in our corpus
num_documents = len(corpus)
idf = {}

# For each term in our document of interest, calculate IDF
for term in set(terms_in_document):
    # Count the number of documents containing the term
    num_documents_containing_term = sum(term in document.split() for document in corpus)
    # Calculate IDF
    idf[term] = np.log(num_documents / (1 + num_documents_containing_term))

# Calculate TF-IDF for each term in the document
tfidf = {term: (tf[term] * idf[term]) for term in terms_in_document}

# Prepare data for histograms
tf_values = list(tf.values())
idf_values = list(idf.values())
tfidf_values = list(tfidf.values())

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# TF Histogram
axes[0].hist(tf_values, bins=len(tf_values), color='skyblue')
axes[0].set_title('Term Frequency (TF) Histogram')
axes[0].set_xlabel('TF Values')
axes[0].set_ylabel('Frequency')

# IDF Histogram
axes[1].hist(idf_values, bins=len(idf_values), color='lightgreen')
axes[1].set_title('Inverse Document Frequency (IDF) Histogram')
axes[1].set_xlabel('IDF Values')
axes[1].set_ylabel('Frequency')

# TF-IDF Histogram
axes[2].hist(tfidf_values, bins=len(tfidf_values), color='salmon')
axes[2].set_title('TF-IDF Histogram')
axes[2].set_xlabel('TF-IDF Values')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

