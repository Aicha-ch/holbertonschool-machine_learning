#!/usr/bin/env python3
"""
Cumulative N-gram BLEU score
"""

import numpy as np
from collections import Counter


def get_ngrams(sentence, n):
    """
    Extracts n-grams from a sentence.
    """
    return [tuple(sentence[i:i+n]) for i in range(len(sentence) - n + 1)]


def count_matches(references, sentence_ngrams, n):
    """
    Counts the number of matching n-grams.
    """
    sentence_ngrams_count = Counter(sentence_ngrams)

    max_ref_ngrams = Counter()
    for reference in references:
        ref_ngrams = get_ngrams(reference, n)
        ref_ngrams_count = Counter(ref_ngrams)
        for ngram in ref_ngrams_count:
            max_ref_ngrams[ngram] = max(
                max_ref_ngrams[ngram], ref_ngrams_count[ngram])

    matches = 0
    for ngram in sentence_ngrams_count:
        matches += min(sentence_ngrams_count[ngram],
                       max_ref_ngrams.get(ngram, 0))

    return matches


def calculate_brevity_penalty(references, sentence_len):
    """
    Calculates brevity penalty.
    """
    closest_ref_len = min((abs(len(ref) - sentence_len), len(ref))
                          for ref in references)[1]

    if sentence_len > closest_ref_len:
        return 1
    else:
        return np.exp(1 - closest_ref_len / sentence_len)


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score.
    """
    precisions = []

    for i in range(1, n + 1):
        sentence_ngrams = get_ngrams(sentence, i)
        matches = count_matches(references, sentence_ngrams, i)

        total_ngrams = len(sentence_ngrams)
        precision = matches / total_ngrams if total_ngrams > 0 else 0
        precisions.append(precision)

    if any(precision == 0 for precision in precisions):
        geometric_mean_precision = 0
    else:
        geometric_mean_precision = np.exp(np.sum(np.log(precisions)) / n)

    brevity_penalty = calculate_brevity_penalty(references, len(sentence))

    return brevity_penalty * geometric_mean_precision
