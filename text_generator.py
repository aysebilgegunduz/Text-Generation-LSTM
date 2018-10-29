"""
# Text Generation using Bidirectional LSTM and Doc2Vec models

The purpose of [this article](https://medium.com/@david.campion/text-generation-using
-bidirectional-lstm-and-doc2vec-models-1-3-8979eb65cb3a) is to discuss about text generation,
using machine learning approaches, especially neural networks.

It is not the first article about it, and probably not the last. Actually, there is a lot of
literature about text generation using "AI" techniques, and some codes are available to generate
texts from existing novels, trying to create new chapters for **"Game of Thrones"**, **"Harry
Potter"**, or a new piece in the style of **Shakespears**. Sometimes with interesting results.

Mainly, these approaches are using classic LSTM networks, and the are pretty fun to be experimented.

However, generated texts provide a taste of unachievement. Generated sentences seems quite right,
whith correct grammar and syntax, as if the neural network was understanding correctly the
structure of a sentence. But the whole new text does not have great sense. If it is not complete
nosense.

This problem could come from the approach itself, using only LSTM to generate text word by word.
But how can we improve them ? In this article, I will try to investigate a new way to generate
sentences.

It does not mean that I will use something completely different from LTSM : I am not, I will use
LTSM network to generate sequences of words. However I will try to go further than a classic LSTM
neural network and I will use an additional neural network (LSTM again), to select the best
phrases.

Then, this article can be used as a tutorial. It describes :

 1. **how to train a neural network to generate sentences** (i.e. sequences of words), based on
 existing novels. I will use a bidirectional LSTM Architecture to perform that.

 2. **how to train a neural network to select the best next sentence for given paragraph** (i.e.
 a sequence of sentences). I will also use a bidirectional LSTM archicture, in addition to a
 Doc2Vec model of the target novels.


### Note about Data inputs

As data inputs, I will not use texts which are not free in term of intellectual properties. So I
will not train the solution to create a new chapter for **"Game of Throne"** or **"Harry Potter"**.
Sorry about that, there is plenty of "free" text to perform such texts generation exercices and
we can dive into the [Gutemberg project](http://www.gutenberg.org), which provides huge amount of
texts (from [William Shakespears](http://www.gutenberg.org/ebooks/author/65) to [H.P. Lovecraft](
http://www.gutenberg.org/ebooks/author/34724), or other great authors).

However, I am also a french author of fantasy and Science fiction. So I will use my personnal
material to create a new chapter of my stories, hoping it can help me in my next work!

So, I will base this exercice on **"Artistes et Phalanges"**, a french fantasy novel I wrote over
the 10 past years, wich I hope will be fair enough in term of data inputs. It contains more than
830 000 charaters.

By the way, if you're a french reader and found of fantasy, you can find it on iBook store and
Amazon Kindle for free... Please note I provide also the data for free on my github repository.
Enjoy it!

## 1. a Neural Network for Generating Sentences

The first step is to generate sentences in the style of a given author.

There is huge literature about it, especially using LSTM to perform such task. As this kind of
network are working well for this job, we will use them.

The purpose of this note is not to deep dive into LSTM description, you can find very great
article about them and I suggest you to read this article
(http://karpathy.github.io/2015/05/21/rnn-effectiveness/) from Andrej Karpathy.

You can also find easily existing code to perform text generation using LSTM. On my github,
you can find two tutorials, one using [Tensorflow](
https://github.com/campdav/text-rnn-tensorflow), and another one using [Keras](
https://github.com/campdav/text-rnn-keras) (over tensorflow), that is easier to understand.

For this first part of these exercice, I will re-use these materials, but with few improvements :

 - Instead of a simple _LSTM_, I will use a _bidirectional LSTM_. This network configuration
 converge faster than a single LSTM (less epochs are required), and from empiric tests,
 seems better in term of accuracy. You can have a look at [this article](
 https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python
 -keras/) from Jason Brownlee, for a good tutorial about bidirectional LSTM.

 - I will use Keras, which require less complexity to create the network of is more readable than
 conventional Tensorflow code.

### 1.1. What is the neural network task in our case ?

LSTM (Long Short Term Memory) are very good for analysing sequences of values and predicting the
next values from them. For example, LSTM could be a very good choice if you want to predict the
very next point of a given time serie (assuming a correlation exist in the sequence).

Talking about sentences and texts ; phrases (sentences) are basically sequences of words. So,
it is natural to assume LSTM could be usefull to generate the next word of a given sentence.

In summary, the objective of a LSTM neural network in this situation is to guess the next word
of a given sentence.

For example:
What is the next word of this following sentence : "he is walking down the"

Our neural net will take the sequence of words as input : "he", "is", "walking", ...
Its ouput will be a matrix providing the probability for each word from the dictionnary to be the
next one of the given sentence.

Then, how will we build the complete text ? Simply iterating the process, by switching the setence
by one word, including the new guessed word at its end. Then, we guess a new word for this new
sentence. ad vitam aeternam.

### 1.1.1. Process

In order to do that, first, we build a dictionary containing all words from the novels we want to
use.

 1. read the data (the novels we want to use),
 1. create the dictionary of words,
 2. create the list of sentences,
 3. create the neural network,
 4. train the neural network,
 5. generate new sentences.

"""

from __future__ import print_function

import os
import numpy as np
import scipy
from pathlib import Path
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from doc2vec import Doc2Vector
from lib.utility_preprocessing import Preprocessing
from lib.seq_embedding import SequenceEmbedding
from lib.sequence_gen import WordPrediction
from keras.models import load_model

DATA_DIR = Path("corpus")
SAVE_DIR = Path('save')  # directory to store models
# file_list = [file.split(".txt")[0] for file in os.listdir("corpus")][:1]
file_list = [DATA_DIR / file for file in os.listdir("corpus")][:1]
vocab_file = SAVE_DIR / "words_vocab-1399.pkl"


class SentenceSelection(object):
    """
    First, we load the library and create the function to define a simple keras Model:

    - bidirectional LSTM,
    - with size of 512 and using RELU as activation (very small, but quicker to perform the test),
    - then a dropout layer of 0,5.
    - The network will not provide me a probability but directly the next vector for a given
    sequence. So I finish it with a simple dense layer of the size of the vector dimension.
    """

    def __init__(self, wordpred, doc2vec, X_train, y_train,
                 nb_sequenced_sentences, do_save=False):
        self.rnn_size = 512  # size of RNN
        self.vector_dim = 500
        self.learning_rate = 0.0001  # learning rate
        self.do_save = do_save
        self.model_sequence = self.bidirectional_lstm_model(nb_sequenced_sentences,
                                                            self.vector_dim)
        self.doc2vec = doc2vec
        self.wordpred = wordpred
        self.wp_model = self.wordpred.model
        self.d2v_model = doc2vec.d2v_model
        self.batch_size = 30  # minibatch size
        self.X_train, self.y_train = X_train, y_train

    def bidirectional_lstm_model(self, seq_length, vector_dim):
        if self.do_save:
            print('Building LSTM model...')
            model = Sequential()
            model.add(Bidirectional(LSTM(self.rnn_size, activation="relu"),
                                    input_shape=(seq_length, vector_dim)))
            model.add(Dropout(0.5))
            model.add(Dense(vector_dim))

            optimizer = Adam(lr=self.learning_rate)
            callbacks = [EarlyStopping(patience=2, monitor='val_loss')]
            model.compile(loss='logcosh', optimizer=optimizer, metrics=['acc'])
            print('LSTM model built.')
        else:
            print("loading sentence selection model...")
            model = load_model(SAVE_DIR + "/" + 'my_model_sequence_lstm.final2.hdf5')
        return model

    def train(self):
        filepath = os.path.join(SAVE_DIR, 'my_model_sequence_lstm.{epoch:02d}.hdf5')
        callbacks = [EarlyStopping(patience=3, monitor='val_loss'),
                     ModelCheckpoint(filepath=filepath,
                                     monitor='val_loss', verbose=1, mode='auto', period=5)]

        self.model_sequence.fit(self.X_train, self.y_train,
                                batch_size=self.batch_size,
                                shuffle=True,
                                epochs=40,
                                callbacks=callbacks,
                                validation_split=0.1)

        # save the model
        self.model_sequence.save(os.path.join(SAVE_DIR, 'my_model_sequence_lstm.final2.hdf5'))

    def create_seed(self, seed_sentences, nb_words_in_seq=20):
        """
        This function is useful to prepare seed sequences, especially if the number of words
        in the seed phrase is lower than the espected number for a sequence.
        """
        # initiate sentences
        generated, sentence = '', []
        # fill the sentence with a default word
        for i in range(nb_words_in_seq):
            sentence.append("le")

        seed = seed_sentences.split()
        for i in range(len(sentence)):
            sentence[nb_words_in_seq - i - 1] = seed[len(seed) - i - 1]
        generated += ' '.join(sentence)
        return generated, sentence

    def define_phrases_candidates(self, sentence, max_words=50, nb_words_in_seq=20,
                                  temperature=1.0, nb_candidates_sents=10):
        """
        the **create_sentences()** function generate a sequence of words (a list) for a
        given spacy doc item.

        It will be used to create a sequence of words from a single phrase.
        """
        phrase_candidate = []
        for i in range(nb_candidates_sents):
            generated_sentence, new_sentence = self.wp_model.predict(
                sentence, max_words=max_words, nb_words_in_seq=nb_words_in_seq,
                temperature=temperature)

            phrase_candidate.append([generated_sentence, new_sentence])

        return phrase_candidate

    def generate_training_vector(self, sentences_list):
        """
        the **generate_training_vector()** function is used to predict the next vectorized-sentence
        for a given sequence of vectorized-sentences.
        :param sentences_list:
        :param verbose:
        :return:
        """
        V = []

        for s in sentences_list:
            # infer the vector of the sentence, from the doc2vec model
            v = self.d2v_model.infer_vector(self.doc2vec.create_sentences(s)[0],
                                            alpha=0.001,
                                            min_alpha=0.001,
                                            steps=10000)
            # create the vector array for the model
            V.append(v)
        V_val = np.array(V)
        del V
        # expand dimension to fit the entry of the model : that's the training vector
        V_val = np.expand_dims(V_val, axis=0)
        return V_val

    def select_next_phrase(self, V_val, candidate_list):
        """
        The **select_next_phrase()** function allows us to pick-up the best candidates for the
        next phrase.

        First, it calculates the vector for each candidates.

        Then, based on the vector generated by the function **generate_training_vector()**, it
        performs a cosine similarity with them and pick the one with the biggest similarity.
        """
        sims_list = []
        # calculate prediction
        preds = self.model_sequence.predict(V_val, verbose=0)[0]

        # calculate vector for each candidate
        for candidate in candidate_list:
            # calculate vector
            # print("calculate vector for : ", candidate[1])
            V = np.array(self.d2v_model.infer_vector(candidate[1]))
            # calculate cosine similarity
            sim = scipy.spatial.distance.cosine(V, preds)
            # populate list of similarities
            sims_list.append(sim)

        # select index of the biggest similarity
        m = max(sims_list)
        index_max = sims_list.index(m)
        return candidate_list[index_max]

    def generate_paragraph(self, phrase_seed, sentences_seed, max_words=50,
                           nb_words_in_seq=20, temperature=1.0, nb_phrases=30,
                           nb_candidates_sents=10):
        """
        The following function, **generate_paragraph()**, combines all previous functions to
        generate the text.

        With the following parameters:
         - phrase_seed : the sentence seed for the first word prediction. It is a list of words.
         - sentences_seed : the seed sequence of sentences. It is a list of sentences.
         - max_words: the maximum number of words for a new generated sentence.
         - nb_words_in_seq: the number of words to keep as seed for the next word prediction.
         - temperature: the temperature for the word prediction.
         - nb_phrases: the number of phrase (sentence) to generate.
         - nb_candidates_sents: the number of phrase candidates to generate for each new phrase.
        """
        sentences_list = sentences_seed
        sentence = phrase_seed
        text = []

        for p in range(nb_phrases):
            print("phrase ", p + 1, "/", nb_phrases)
            # generate seed training vector
            V_val = self.generate_training_vector(sentences_list)

            # generate phrase candidate
            phrases_candidates = self.define_phrases_candidates(
                sentence,
                max_words=max_words,
                nb_words_in_seq=nb_words_in_seq,
                temperature=temperature,
                nb_candidates_sents=nb_candidates_sents)

            next_phrase = self.select_next_phrase(V_val, phrases_candidates)

            print("Next phrase: ", next_phrase[0])

            for i in range(len(sentences_list) - 1):
                sentences_list[i] = sentences_list[i + 1]

            sentences_list[len(sentences_list) - 1] = next_phrase[0]
            sentence = next_phrase[1]
            text.append(next_phrase[0])

        return text


if __name__ == '__main__':
    save_embedding = True
    preprocessing = Preprocessing(file_list, vocab_file, do_save=False)

    sequence_embed = SequenceEmbedding(preprocessing.vocab, preprocessing.wordlist,
                                       preprocessing.tokenlist, preprocessing.vocab_size,
                                       do_save=True)
    if sequence_embed.do_save:
        sequence_embed.word_embedding()
        sequence_embed.train()
    else:
        sequence_embed.load_saved_model()
    sequence_embed.model.summary()

    if save_embedding:
        sequence_embed.write_embeddings(
            SAVE_DIR / "embedding.csv", {},
            sequence_embed.model.get_layer("target-embedding").get_weights()[0])

    X_phrase, y_phrase = sequence_embed.create_training_phrase(SAVE_DIR / "embedding.csv")

    word_pred = WordPrediction(X_phrase, y_phrase, sequence_embed, do_save=True)
    if word_pred.do_save:
        word_pred.bidirectional_lstm_model(word_pred.seq_length, word_pred.vocab_size)
        word_pred.train()
    else:
        word_pred.get_saved_model()
    word_pred.model.summary()

    # TODO: Understand below code
    doc2vec = Doc2Vector(file_list, do_save=True)
    X_train, y_train = doc2vec.doc2vec()
    print(X_train.shape, y_train.shape)

    sentences_list = [
        'Happy families are all alike; every unhappy family is unhappy in its own way.',
        'Everything was in confusion in the Oblonskys’ house.',
        'The wife had discovered that the husband was carrying on an intrigue with a French girl, '
        'who had been a governess in their family, and she had announced to her husband that she '
        'could not go on living in the same house with him.',
        'This position of affairs had now lasted three days, and not only the husband and wife '
        'themselves, but all the members of their family and household, were painfully conscious '
        'of it.',
        'Every person in the house felt that there was no sense in their living together, '
        'and that the stray people brought together by chance in any inn had more in common with '
        'one another than they, the members of the family and household of the Oblonskys.',
        'The wife did not leave her own room, the husband had not been at home for three days.',
        'The children ran wild all over the house; the English governess quarreled with the '
        'housekeeper, and wrote to a friend asking her to look out for a new situation for her; '
        'the man-cook had walked off the day before just at dinner time; the kitchen-maid, '
        'and the coachman had given warning.',
        'Three days after the quarrel, Prince Stepan Arkadyevitch Oblonsky—Stiva, as he was '
        'called in the fashionable world—woke up at his usual hour, that is, at eight o’clock in '
        'the morning, not in his wife’s bedroom, but on the leather-covered sofa in his study.',
        'He turned over his stout, well-cared-for person on the springy sofa, as though he would '
        'sink into a long sleep again; he vigorously embraced the pillow on the other side and '
        'buried his face in it; but all at once he jumped up, sat up on the sofa, and opened his '
        'eyes.',
        '"Yes, yes, how was it now?"',
        'he thought, going over his dream.',
        '"Now, how was it?',
        'To be sure!',
        'Alabin was giving a dinner at Darmstadt; no, not Darmstadt, but something American.',
        'Yes, but then, Darmstadt was in America.']

    sentence_p = SentenceSelection(word_pred, doc2vec, X_train, y_train, 15, do_save=True)
    phrase_seed, sentences_seed = sentence_p.create_seed(" ".join(sentences_list), 20)
    print(phrase_seed)
    print(sentences_seed)

    text = sentence_p.generate_paragraph(sentences_seed,
                                         sentences_list,
                                         max_words=80,
                                         nb_words_in_seq=30,
                                         temperature=0.201,
                                         nb_phrases=5,
                                         nb_candidates_sents=7)

    print("generated text: ")
    for t in text:
        print(t)
