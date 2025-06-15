#!/usr/bin/env python3
"""
Semantic search
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def semantic_search(corpus_path, sentence):
    """
    Perform semantic search on a corpus of documents.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    file_names = os.listdir(corpus_path)
    for file_name in file_names:
        if file_name.endswith('.md'):
            file_path = os.path.join(corpus_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    doc_embeddings = model.encode(documents)

    query_embedding = model.encode([sentence])[0]

    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    best_doc_index = np.argmax(similarities)

    return documents[best_doc_index]
