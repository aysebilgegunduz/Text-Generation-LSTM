from __future__ import print_function

import os
from pathlib import Path
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

DATA_DIR = Path("corpus")
SAVE_DIR = Path('save')  # directory to store models


class WordPrediction(object):
    """
    Now, here come the fun part. The creation of the neural network.
    As you will see, I am using Keras which provide very good abstraction to design an architecture.

    In this example, I create the following neural network:
     - bidirectional LSTM,
     - with size of 256 and using RELU as activation,
     - then a dropout layer of 0,6 (it's pretty high, but necesseray to avoid quick divergence)


    The net should provide me a probability for each word of the vocabulary to be the next one
    after a given sentence. So I end it with:

     - a simple dense layer of the size of the vocabulary,
     - a softmax activation.

    I use ADAM as optimizer and the loss calculation is done on the categorical cross-entropy.

    Here is the function to build the network:"""

    def __init__(self, features, labels, sequence_emb, do_save=False):
        self.features = features
        self.labels = labels
        self.sequence_emb = sequence_emb
        self.vocab_size = self.sequence_emb.vocab_size
        self.rnn_size = 256  # size of RNN
        self.batch_size = 32  # minibatch size
        self.seq_length = 30  # sequence length
        self.num_epochs = 50  # number of epochs
        self.learning_rate = 0.001  # learning rate
        self.sequences_step = 1  # step to create sequences
        self.do_save = do_save
        self.model = None

    def bidirectional_lstm_model(self, seq_length, vocab_size):
        print('Build LSTM model.')
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(self.rnn_size, activation="relu"),
                                     input_shape=(seq_length, vocab_size)))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(vocab_size))
        self.model.add(Activation('softmax'))

        optimizer = Adam(lr=self.learning_rate)
        callbacks = [EarlyStopping(patience=2, monitor='val_loss')]
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=[categorical_accuracy])

    def get_saved_model(self):
        print("loading word prediction model...")
        self.model = load_model(SAVE_DIR + "/" + 'my_model_gen_sentences_lstm.final.hdf5')

    def batch_generator(self):
        pass

    def train(self):
        # fit the model
        filepath = SAVE_DIR / 'my_model_gen_sentences_lstm.{epoch:02d}-{val_loss:.2f}.hdf5'
        callbacks = [EarlyStopping(patience=4, monitor='val_loss'),
                     ModelCheckpoint(filepath=filepath,
                                     monitor='val_loss', verbose=0,
                                     mode='auto', period=2)]
        # batch_generator = self.batch_generator()

        self.model.fit(self.features, self.labels,
                       batch_size=self.batch_size,
                       shuffle=True,
                       epochs=self.num_epochs,
                       callbacks=callbacks,
                       validation_split=0.01)

    def predict(self, seed_sentences, max_words=50, nb_words_in_seq=20, temperature=1.0):
        """
        to predict the next word of a given sequence of words. In order to generate text, the
        task is pretty simple:

         - we define a "seed" sequence of 30 words (30 is the number of words required by the
         neural net for the sequences),
         - we ask the neural net to predict word number 31,
         - then we update the sequence by moving words by a step of 1, adding words number
         31 at its end,
         - we ask the neural net to predict word number 32,
         - etc. For as long as we want.

        Doing this, we generate phrases, word by word.
        """
        sentences = ['a' for i in range(self.seq_length)]
        seed = seed_sentences.split()
        for i in range(len(seed)):
            sentences[self.seq_length - i - 1] = seed[len(seed) - i - 1]

        generated = ' '.join(sentences)
        print('Generating text with the following seed: "' + ' '.join(sentences) + '"')

        for i in range(max_words):
            x = self.sequence_emb.create_embedding(sentences, nb_words_in_seq,
                                                   SAVE_DIR / "embedding.csv")

            # calculate next word
            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sequence_emb.sample(preds, temperature)
            next_word = self.sequence_emb.wordlist[next_index]

            # add the next word to the text
            generated += " " + next_word
            # shift the sentences by one, and and the next word at its end
            sentences = sentences[1:] + [next_word]

        return generated, sentences
