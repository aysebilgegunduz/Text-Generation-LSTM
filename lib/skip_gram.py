from __future__ import print_function
import logging
import numpy as np
# from numba import jit

RAND_MAX = 2147483647


# @jit(nopython=True)
def skip_gram_iterator(sequence, window_size, negative_samples, seed):
    """ An iterator which at each step returns a tuple of (word, context, label) """

    sequence_length = sequence.shape[0]
    epoch = 0
    i = 0
    while True:
        window_start = max(0, i - window_size)
        window_end = min(sequence_length, i + window_size + 1)
        for j in range(window_start, window_end):
            if i != j:
                yield (sequence[i], sequence[j], 1)

        for _ in range(negative_samples):
            random_float = np.random.rand()
            j = int(random_float * sequence_length)
            yield (sequence[i], sequence[j], 0)

        i += 1
        if i == sequence_length:
            epoch += 1
            logging.info("iterated %d times over data set", epoch)
            i = 0


# @jit(nopython=True)
def batch_iterator(sequence, window_size, negative_samples, batch_size, seed=1):
    """ An iterator which returns training instances in batches """
    iterator = skip_gram_iterator(sequence, window_size, negative_samples, seed)
    words = np.empty(shape=batch_size, dtype=np.int64)
    contexts = np.empty(shape=batch_size, dtype=np.int64)
    labels = np.empty(shape=batch_size, dtype=np.int64)
    while True:
        for i in range(batch_size):
            word, context, label = next(iterator)
            words[i] = word
            contexts[i] = context
            labels[i] = label
        yield ([words, contexts], labels)
