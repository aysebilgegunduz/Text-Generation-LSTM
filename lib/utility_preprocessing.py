from __future__ import print_function

import os
import codecs
from ktext.preprocess import processor
from six.moves import cPickle
from nltk.tokenize import sent_tokenize


class Preprocessing(object):
    def __init__(self, file_list, vocab_file, hueristic_pct_padding=0.7,
                 keep_n=30000, do_save=False):
        """
        word_counts = collections.Counter(wordlist)

        # Mapping from index to word : that's the vocabulary
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))

        # Mapping from word to index
        vocab = {x: i for i, x in enumerate(vocabulary_inv)}
        words = [x[0] for x in word_counts.most_common()]
        """
        self.code_proc = processor(hueristic_pct_padding=hueristic_pct_padding,
                                   keep_n=keep_n)
        self.VOCAB_FILE = vocab_file
        self.FILE_LIST = file_list
        self.t_code = None
        self.vocab = {}
        self.wordlist = []
        self.tokenlist = []
        if do_save:
            self.preprocess_token(do_save)
        else:
            self.load_vocab_file()
        self.vocab_size = len(self.vocab.keys())

    def read_words(self, file_lists):
        for filename in file_lists:
            with codecs.open(filename, "r") as f:
                yield f.read()

    def build_vocab(self, file_lists, top=800000):
        """
        words_map => word to index
        corpus => words list
        :param file_lists:
        :param top:
        :return:
        """
        texts = [text for text in self.read_words(file_lists)]
        padding_maxlen = 0
        sentences = []
        for text in texts:
            sentences.extend(sent_tokenize(text))
            padding_maxlen = max(padding_maxlen, max(len(sent) for sent in sentences))
        body_pp = processor(keep_n=top, padding_maxlen=padding_maxlen)
        train_body_vecs = body_pp.fit_transform(sentences)

        return body_pp, train_body_vecs

    def index2words(self, id2token, indices):
        """ Give a index and words tuple
        """
        return [id2token[index] for index in indices if index in id2token]

    def preprocess_token(self, do_save=False):
        """
        word_map: Mapping from index to word : that's the vocabulary
        vocab: words -> index
        words = words
        :return:
        """
        self.__clean()
        body_pp, train_body_vecs = self.build_vocab(self.FILE_LIST)
        self.vocab = body_pp.token2id
        id2token = body_pp.id2token
        for indices in train_body_vecs:
            self.wordlist.append(self.index2words(id2token, indices))
        self.tokenlist = train_body_vecs

        if do_save:
            self.save_vocab_file()

    def __clean(self):
        self.tokenlist = []
        self.wordlist = []
        self.vocab = {}

    def save_vocab_file(self):
        """
        save the words and vocabulary
        :return:
        """
        with open(os.path.join(self.VOCAB_FILE), 'wb') as f:
            cPickle.dump((self.tokenlist, self.wordlist, self.vocab), f)

    def load_vocab_file(self):
        with open(os.path.join(self.VOCAB_FILE), 'rb') as f:
            self.tokenlist, self.wordlist, self.vocab = cPickle.load(f)