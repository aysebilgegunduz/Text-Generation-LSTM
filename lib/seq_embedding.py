from __future__ import print_function

import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, merge, Input, Reshape, Flatten
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.preprocessing import sequence
# from keras.preprocessing.sequence import skipgrams
from lib import skip_gram
from pathlib import Path
from itertools import chain

DATA_DIR = Path("corpus")
SAVE_DIR = Path('save')  # directory to store models


class SequenceEmbedding(object):
    """
    Create Sequences
    ====================

    Now, we have to create the input data for our LSTM. We create two lists:

    - **sequences**: this list will contain the sequences of words used to train the model,
    - **next_words**: this list will contain the next words for each sequences of
    the **sequences** list.

    In this exercice, we assume we will train the network with sequences of 30 words
    (seq_length = 30).

    So, to create the first sequence of words, we take the 30th first words in the
    **wordlist** list. The word 31 is the next word of this first sequence, and is added
    to the **next_words** list.

    Then we jump by a step of 1 (sequences_step = 1 in our example) in the list of words,
    to create the second sequence of words and retrieve the second "next word".

    We iterate this task until the end of the list of words.

    """

    def __init__(self, vocab, wordlist, tokenlist, vocab_size, seq_length=30,
                 sequences_step=1, epochs=100, batch_size=128, window_size=3,
                 do_save=False):
        self.model_checkpoint = None
        self.validation_model = None
        self.vocab = vocab  # Dict[str, int]
        self.wordlist = wordlist  # List[List[str]]
        self.tokenlist = tokenlist  # List[List[int]
        self.vocab_size = vocab_size
        self.seq_length = seq_length  # sequence length
        self.sequences_step = sequences_step  # step to create sequences
        # arbitrarily set latent dimension for embedding and hidden units
        self.latent_dim = 300
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.do_save = do_save
        self.sequences, self.next_words = [], []
        self.model = None

    def generate_sequence(self):
        """ Generate sequence of tokens """
        sequences = []
        next_words = []
        for doc in self.wordlist:
            for j in range(0, len(doc) - self.seq_length, self.sequences_step):
                sequences.append(self.tokenlist[doc][j: j + self.seq_length])
                next_words.append(self.tokenlist[doc][j + self.seq_length])
        return sequences, next_words

    def batch_iterator(self, word_target, word_context, words_labels, batch_size):
        """ An iterator which returns training instances in batches """
        words = np.empty(shape=batch_size, dtype=np.int64)
        contexts = np.empty(shape=batch_size, dtype=np.int64)
        labels = np.empty(shape=batch_size, dtype=np.int64)
        while True:
            for i in range(batch_size):
                idx = np.random.randint(0, len(labels) - 1)
                word, context, label = word_target[idx], word_context[idx], words_labels[idx]
                words[i] = word
                contexts[i] = context
                labels[i] = label
            yield ([words, contexts], labels)

    def word_embedding(self):
        input_target = Input((1,))
        input_context = Input((1,))

        target = Embedding(self.vocab_size, self.latent_dim, name='target-embedding',
                           mask_zero=False)(input_target)
        context = Embedding(self.vocab_size, self.latent_dim, name="context-embedding",
                            mask_zero=False)(input_context)
        dot_product = merge.dot([target, context], axes=2, normalize=False, name="dot")
        # dot_product = Reshape((1,))(dot_product)
        dot_product = Flatten()(dot_product)

        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid', name='output')(dot_product)

        # setup a cosine similarity operation which will be output in a secondary model
        # similarity = merge.dot([target, context], axes=2, normalize=True)

        # create the primary training model
        self.model = Model(input=[input_target, input_context], output=output)
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        # create a secondary validation model to run our similarity checks during training
        # self.validation_model = Model(input=[input_target, input_context], output=similarity)
        self.model_checkpoint = ModelCheckpoint(
            filepath=SAVE_DIR / 'word2vect-model-{epoch:02d}.hdf5',
            verbose=1, save_best_only=True)

    def train(self):
        negative_samples = 1
        tokenlist = np.array(self.flatten(self.tokenlist), dtype=np.int)
        seq_length = tokenlist.shape[0]
        approx_steps_per_epoch = (seq_length * (self.window_size * 2.0) +
                                  seq_length * negative_samples) / self.batch_size

        batch_iterator = skip_gram.batch_iterator(tokenlist, self.window_size,
                                                  negative_samples, self.batch_size)

        self.model.fit_generator(batch_iterator,
                                 steps_per_epoch=approx_steps_per_epoch,
                                 epochs=self.epochs,
                                 verbose=True,
                                 max_queue_size=100,
                                 callbacks=[self.model_checkpoint])

        self.model.save(SAVE_DIR / 'word2vec-model.h5')

    def flatten(self, tokenlist):
        return list(chain.from_iterable(tokenlist))

    def load_saved_model(self):
        self.model = load_model(SAVE_DIR / 'word2vec-model.h5')

    def write_embeddings(self, path, index2word, embeddings):
        # logging.info("Saving embeddings to %s", path)
        np_index = np.empty(shape=len(index2word), dtype=object)
        for index, word in index2word.items():
            np_index[index] = word
        df = pd.DataFrame(data=embeddings, index=np_index)
        df.to_csv(path, float_format="%.4f", header=False)

    def create_training_phrase(self, path):
        embeddings = pd.read_csv(path, index_col=0)
        sequences, next_words = self.generate_sequence()
        nb_sequences = len(sequences)
        X = np.zeros((nb_sequences, self.seq_length, self.latent_dim), dtype=np.int64)
        y = np.zeros((nb_sequences, self.latent_dim), dtype=np.int64)
        for i, sentence in enumerate(sequences):
            for t, index in enumerate(sentence):
                X[i, t, :] = embeddings.loc[index]
            y[i, next_words[i]] = embeddings.loc[i]

        return X, y

    def create_embedding(self, sequences, seq_length, path):
        embeddings = pd.read_csv(path, index_col=0)
        nb_sequences = len(sequences)
        X = np.zeros((nb_sequences, seq_length, self.latent_dim), dtype=np.int64)
        for i, sentence in enumerate(sequences):
            for t, index in enumerate(sentence):
                X[i, t, :] = embeddings.loc[index]

        return X

    def sample(self, preds, temperature=1.0):
        """
        To improve the text generation, and tune a bit the word prediction, we introduce a
        specific function to pick-up words from our vocabulary.

        We will not take the words with the highest prediction (or the generation of text will be
        boring), but would like to insert some uncertainties, and let the solution, sometime,
        to pick-up words with less good prediction.

        That is the purpose of the function **sample()**, that will draw randomly a word from our
        vocabulary.

        However, the probability for a word to be drawn will depends directly on its probability
        to be the next word, thanks to our first bidirectional LSTM Model.

        In order to tune this probability, we introduce a "temperature" to smooth or sharpen its
        value.

         - **if _temperature = 1.0_**, the probability for a word to be drawn is equal to the
         probability for the word to be the next one in the sequence (output of the owrd
         prediction model),

         - **if _temperature_ is big (much bigger than 1)**, the range of probabilities is
         shorten: the probabilities for all words to be the next one is closer to 1. More variety
         of words will be picked-up from the vocabulary.

         - **if _temperatune_ is small (close to 0)**, small probabilities will be avoided (they
         will be set closed to 0). Less words will be picked-up from the vocabulary.

        """
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
