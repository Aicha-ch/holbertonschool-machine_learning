#!/usr/bin/env python3
"""
Question Answering
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference 
    document to answer a question.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

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
