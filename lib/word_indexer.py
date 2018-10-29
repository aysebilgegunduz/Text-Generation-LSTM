from __future__ import print_function

import collections
import math
from copy import deepcopy
from joblib import Parallel, delayed, cpu_count, my_exceptions
from itertools import groupby, chain
from more_itertools import chunked
import multiprocessing

COUNT = [['UNK', -1]]
_GLOBAL_CHUNK_SIZE = 1000


class Counter(object):
    def __init__(self, initval=0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        return self.val.value


def flatten(list2d):
    return list(chain.from_iterable(*list2d))


def map_index(dictionary, words, unk_count):
    indices = []
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count.increment()
        indices.append(index)
    return indices


def reduce_words(words_map, sequence, n_words):
    unk_count = Counter(0)
    manager = multiprocessing.Manager()
    chunk_size = _GLOBAL_CHUNK_SIZE
    allwords = []
    for key, group in groupby(words_map, lambda x: x[0]):
        total_count = sum([thing[1] for thing in group])
        allwords.append((key, total_count))

    allwords.sort(key=lambda x: x[1])
    count = deepcopy(COUNT)
    count.extend(allwords[:n_words])

    dictionary = {pair[0]: index for index, pair in enumerate(count)}
    shared_dict = manager.dict(dictionary)

    indices = Parallel(n_jobs=-1)(delayed(map_index)(shared_dict, chunk, unk_count)
                                  for chunk in chunked(sequence, chunk_size))
    count[0][1] = unk_count.value
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return flatten(indices), count, dictionary, reversed_dictionary


def build_dataset(words):
    return collections.Counter(words).most_common()


def preprocessing(sequence, n_words):
    cpu_cores = cpu_count()
    indices, count, dictionary, reversed_dictionary = None, None, None, None
    try:
        chunk_size = math.ceil(len(sequence) / cpu_cores)
        chunk_size = _GLOBAL_CHUNK_SIZE
        map_words = Parallel(n_jobs=-1)(delayed(build_dataset)(word) for word in
                                        chunked(sequence, chunk_size))
        map_words = flatten(map_words)
        indices, count, dictionary, reversed_dictionary = reduce_words(map_words,
                                                                       sequence,
                                                                       n_words)

    except my_exceptions.JoblibException as e:
        print("Exception occur while indexing", e)
        raise e
    finally:
        return indices, count, dictionary, reversed_dictionary
