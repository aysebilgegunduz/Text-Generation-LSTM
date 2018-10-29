from __future__ import print_function

import codecs
import time
import os
import gensim
import numpy as np
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer
from six.moves import cPickle
from pathlib import Path

DATA_DIR = Path("corpus")
SAVE_DIR = Path('save')  # directory to store models


class Doc2Vector(object):
    """
    # Text Generation using Bidirectional LSTM and Doc2Vec models 2/3

    If you have reached directly this page, I suggest to start reading the first part of this
    article. It describes how to create a RNN model to generate a text, word after word.

    I finished the first part of the article explaining I will try to improve the generation
    of sentences, by detecting patterns in the sequences of sentences, not only in the
    sequences of words.

    It could be an improvement, because doing that, the context of a paragraph (is it a
    description of a countryside? a dialog between characters? which people are involved?
    what are the previous actions? etc.) could emerge and can be used to select wisely the
    next sentence of the text.

    The process will be similar to the previous one, however, I will have to vectorize all
    sentences in the text, and try to find patterns in sequences of these vectors.

    In order to do that, we will use **Doc2Vec**.

    Doc2Vec

    Doc2Vec is able to vectorize a paragraph of text. If you do not know it, I suggest to
    have a look on the gensim web site, that describes how its work and what you’re allowed
    to do with it.

    In a short, we will transform each sentences of our text in a vector of a specific space.
    The great thing of the approach is we will be able to compare them ; by example,
    to retrieve the most similar sentence of a given one.

    Last but not least, the dimension of the vectors will be the same, whatever is the number
    of words in their linked sentence.

    It is exactly what we are looking for: I will be able to train a new LSTM, trying to
    catch pattern from sequences of vectors of the same dimensions.

    I have to be honest: I am not sure we can perform such task with enough accuracy,
    but let’s have some tests. It is an experiment, at worst, it will be a good exercice.``

    So, once all sentences will be converted to vectors, we will try to **train a new
    bidirectional LSTM**. It purpose will be to predict the best vector, next to a sequence
    of vectors.

    Then how will we generate text ?

    Pretty easy: thanks to our previous LSTM model, we will generate sentences as candidates
    to be the next phrase. We will infer their vectors using the **trained doc2Vec model**,
    then pick the closest one to the prediction of our new LSTM model.

    """

    def __init__(self, file_list, do_save=False):
        self.FILE_LIST = file_list
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.do_save = do_save
        self.d2v_model = None

    def save_sentence_vector(self, d2v_model, sentences, do_save=True):
        if do_save:
            sentences_vector = []
            t = 500
            for i, sent in enumerate(sentences):
                if i % t == 0:
                    print("sentence-{0} : {1} \n ***** ".format(i, sent))
                sentences_vector.append(d2v_model.infer_vector(sent, alpha=0.001,
                                                               min_alpha=0.001,
                                                               steps=10000))

            # save the sentences_vector
            sentences_vector_file = SAVE_DIR / 'sentences_vector_500_a001_ma001_s10000.pkl'
            with open(os.path.join(sentences_vector_file), 'wb') as f:
                cPickle.dump(sentences_vector, f)
        else:
            sentences_vector_file = SAVE_DIR / 'sentences_vector_500_a001_ma001_s10000.pkl'
            with open(os.path.join(sentences_vector_file), 'rb') as f:
                sentences_vector = cPickle.load(f)

        return sentences_vector

    def create_sentences(self):
        """ create sentences from files """
        sentences = []
        for file_name in self.FILE_LIST:
            input_file = DATA_DIR / file_name

            with codecs.open(input_file, "r") as f:
                data = f.read()

            # use NLTK sentence tokenizer
            doc = self.sent_tokenizer.tokenize(data)
            sents = [self.word_tokenizer.tokenize(sent) for sent in doc]
            sentences.append(sents)
            # sentences = sentences + sents

        return sentences

    def doc2vec(self, do_save=False):
        sentences_label = []
        sentences = self.create_sentences()

        # create labels
        for i in range(np.array(sentences).shape[0]):
            sentences_label.append("ID" + str(i))

        # Here are some insights for the used parameters:
        #
        # - **dimensions**: 300 dimensions seem to work well for classic subjects. In my case,
        # after few tests, I prefer to choose 500 dimensions,

        # - **epochs**: below 10 epochs, results are not good enough (similarity are not working
        # well), and bigger number of epochs creates to much similar vectors. So I chose 20
        # epochs for the training.

        # - **min_count**: I want to integrate all words in the training, even those with very
        # few occurrence. Indeed, I assume that, for my exercice, specific words could be
        # important. I set the value to 0, but 3 to 5 should be OK.

        # - **sample**: *0.0*. I do not want to downsample randomly higher-frequency words,
        # so I disabled it.

        # - **hs and dm**: Each time I want to infer a new vector from the trained model,
        # for a given sentence, I want to have the same output vector. In order to do that (
        # strangely it’s not so intuitive), I need to use a distributed bag of words as *training
        # algorithm (dm=0)* and *hierarchical softmax (hs=1)*. Indeed, for my purpose,
        # distributive memory and negative sampling seems to give less good results."""

        if self.do_save:
            self.train_doc2vec_model(sentences, sentences_label, size=500, sample=0.0,
                                     alpha=0.025, min_alpha=0.001, min_count=0,
                                     window=10, epoch=20, dm=0, hs=1,
                                     save_file=DATA_DIR / 'doc2vec.w2v')

        # load the model
        self.d2v_model = gensim.models.doc2vec.Doc2Vec.load(DATA_DIR / 'doc2vec.w2v')
        sentences_vector = self.save_sentence_vector(self.d2v_model, sentences, do_save)
        nb_sequenced_sentences = 15
        vector_dim = 500

        X_train = np.zeros((len(sentences), nb_sequenced_sentences, vector_dim), dtype=np.float)
        y_train = np.zeros((len(sentences), vector_dim), dtype=np.float)

        t = 1000
        for i in range(len(sentences_label) - nb_sequenced_sentences - 1):
            if i % t == 0:
                print("new sequence: ", i)

            for k in range(nb_sequenced_sentences):
                sent = sentences_label[i + k]
                vect = sentences_vector[i + k]

                if i % t == 0:
                    print("  ", k + 1, "th vector for this sequence. Sentence ", sent,
                          "(vector dim = ", len(vect), ")")

                for j in range(len(vect)):
                    X_train[i, k, j] = vect[j]

            senty = sentences_label[i + nb_sequenced_sentences]
            vecty = sentences_vector[i + nb_sequenced_sentences]
            if i % t == 0:
                print("  y vector for this sequence ", senty, ": (vector dim = ", len(vecty), ")")
            for j in range(len(vecty)):
                y_train[i, j] = vecty[j]

        return X_train, y_train

    def train_doc2vec_model(self, data, docLabels, size=300, sample=0.000001, dm=0, hs=1,
                            window=10, min_count=0, workers=8, alpha=0.024, min_alpha=0.024,
                            epoch=15, save_file='doc2vec.w2v'):
        """ Train doc to vec model """

        class LabeledLineSentence(object):
            def __init__(self, doc_list, labels_list):
                self.labels_list = labels_list
                self.doc_list = doc_list

            def __iter__(self):
                for idx, doc in enumerate(self.doc_list):
                    yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])

        startime = time.time()

        print("{0} articles loaded for model".format(len(data)))

        it = LabeledLineSentence(data, docLabels)

        model = gensim.models.Doc2Vec(size=size, sample=sample, dm=dm, window=window,
                                      min_count=min_count, workers=workers, alpha=alpha,
                                      min_alpha=min_alpha, hs=hs)  # use fixed learning rate
        model.build_vocab(it)
        for epoch in range(epoch):
            print("Training epoch {}".format(epoch + 1))
            model.train(it, total_examples=model.corpus_count, epochs=model.iter)
            # model.alpha -= 0.002 # decrease the learning rate
            # model.min_alpha = model.alpha # fix the learning rate, no decay

        # saving the created model

        model.save(os.path.join(save_file))
        print('model saved')