#!/usr/bin/env python3
"""
Multi-reference Question Answering
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
sbert = SentenceTransformer('all-MiniLM-L6-v2')


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

    documents = []
    file_names = os.listdir(corpus_path)
    for file_name in file_names:
        if file_name.endswith('.md'):
            file_path = os.path.join(corpus_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    doc_embeddings = sbert.encode(documents)

    query_embedding = sbert.encode([sentence])[0]

    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    best_doc_index = np.argmax(similarities)

    return documents[best_doc_index]


def find_answer(question, reference):
    """
    Finds a snippet of text within a reference 
    document to answer a question.
    """
    inputs = tokenizer(question, reference, return_tensors="tf")

    input_tensors = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    output = model(input_tensors)

    start_logits = output[0]
    end_logits = output[1]

    sequence_length = inputs["input_ids"].shape[1]

    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    if not answer.strip():
        return None

    return answer


def question_answer(corpus_path):
    """
    questio answer
    """

    farewells = ["exit", "quit", "goodbye", "bye"]

    user_input = ""
    while True:
        user_input = input("Q: ")

        if user_input.lower() in farewells:
            print("A: Goodbye")
            return

        best_document = semantic_search(corpus_path, user_input)

        answer = find_answer(question=user_input, reference=best_document)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
