#!/usr/bin/env python3
"""
Tokenize tensorflow Dataset
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset object, loads the training and validation
        datasets, and sets up the data pipeline.
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
                self.data_train)

        # Set up the training data pipeline
        self.data_train = self.data_train.map(
                self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_train = self.data_train.filter(
                lambda pt,
                en: tf.logical_and(
                    tf.size(pt) <= max_len, tf.size(en) <= max_len)
                )
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=20000)
        self.data_train = self.data_train.padded_batch(
                batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
                buffer_size=tf.data.AUTOTUNE)

        # Set up the validation data pipeline
        self.data_valid = self.data_valid.map(
                self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = self.data_valid.filter(
                lambda pt,
                en: tf.logical_and(
                    tf.size(pt) <= max_len, tf.size(en) <= max_len))
        self.data_valid = self.data_valid.padded_batch(
                batch_size, padded_shapes=([None], [None])
                )

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers and adapts them to
        the datatset.
        """

        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
                'neuralmind/bert-base-portuguese-cased', use_fast=True,
                clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
                'bert-base-uncased', use_fast=True,
                clean_up_tokenization_spaces=True)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation pair into tokenized sentences.
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + \
                    [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + \
                    [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        A TensorFlow wrapper.
        """
        pt_tokens, en_tokens = tf.py_function(func=self.encode,
                                              inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens