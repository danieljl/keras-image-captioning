from keras.applications.inception_v3 import InceptionV3
from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import (Dense, Embedding, GRU, Input, LSTM, RepeatVector,
                          TimeDistributed)
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l1_l2

from .config import active_config
from .losses import categorical_crossentropy_from_logits
from .metrics import categorical_accuracy_with_variable_timestep
from .word_vectors import get_word_vector_class


class ImageCaptioningModel(object):
    """A very configurable model to produce captions from images."""

    def __init__(self,
                 learning_rate=None,
                 vocab_size=None,
                 embedding_size=None,
                 rnn_output_size=None,
                 dropout_rate=None,
                 bidirectional_rnn=None,
                 rnn_type=None,
                 rnn_layers=None,
                 l1_reg=None,
                 l2_reg=None,
                 initializer=None,
                 word_vector_init=None):
        """
        If an arg is None, it will get its value from config.active_config.
        """
        self._learning_rate = learning_rate or active_config().learning_rate
        self._vocab_size = vocab_size or active_config().vocab_size
        self._embedding_size = embedding_size or active_config().embedding_size
        self._rnn_output_size = (rnn_output_size or
                                 active_config().rnn_output_size)
        self._dropout_rate = dropout_rate or active_config().dropout_rate
        self._rnn_type = rnn_type or active_config().rnn_type
        self._rnn_layers = rnn_layers or active_config().rnn_layers
        self._word_vector_init = (word_vector_init or
                                  active_config().word_vector_init)

        self._initializer = initializer or active_config().initializer
        if self._initializer == 'vinyals_uniform':
            self._initializer = RandomUniform(-0.08, 0.08)

        if bidirectional_rnn is None:
            self._bidirectional_rnn = active_config().bidirectional_rnn
        else:
            self._bidirectional_rnn = bidirectional_rnn

        l1_reg = l1_reg or active_config().l1_reg
        l2_reg = l2_reg or active_config().l2_reg
        self._regularizer = l1_l2(l1_reg, l2_reg)

        self._keras_model = None

        if self._vocab_size is None:
            raise ValueError('config.active_config().vocab_size cannot be '
                             'None! You should check your config or you can '
                             'explicitly pass the vocab_size argument.')

        if self._rnn_type not in ('lstm', 'gru'):
            raise ValueError('rnn_type must be either "lstm" or "gru"!')

        if self._rnn_layers < 1:
            raise ValueError('rnn_layers must be >= 1!')

        if self._word_vector_init is not None and self._embedding_size != 300:
            raise ValueError('If word_vector_init is not None, embedding_size '
                             'must be 300')

    @property
    def keras_model(self):
        if self._keras_model is None:
            raise AttributeError('Execute build method first before accessing '
                                 'keras_model!')
        return self._keras_model

    def build(self, vocabs=None):
        if self._keras_model:
            return
        if vocabs is None and self._word_vector_init is not None:
            raise ValueError('If word_vector_init is not None, build method '
                             'must be called with vocabs that are not None!')

        image_input, image_embedding = self._build_image_embedding()
        sentence_input, word_embedding = self._build_word_embedding(vocabs)
        sequence_input = Concatenate(axis=1)([image_embedding, word_embedding])
        sequence_output = self._build_sequence_model(sequence_input)

        model = Model(inputs=[image_input, sentence_input],
                      outputs=sequence_output)
        model.compile(optimizer=Adam(lr=self._learning_rate, clipnorm=5.0),
                      loss=categorical_crossentropy_from_logits,
                      metrics=[categorical_accuracy_with_variable_timestep])

        self._keras_model = model

    def _build_image_embedding(self):
        image_model = InceptionV3(include_top=False, weights='imagenet',
                                  pooling='avg')
        for layer in image_model.layers:
            layer.trainable = False

        dense_input = BatchNormalization(axis=-1)(image_model.output)
        image_dense = Dense(units=self._embedding_size,
                            kernel_regularizer=self._regularizer,
                            kernel_initializer=self._initializer
                            )(dense_input)
        # Add timestep dimension
        image_embedding = RepeatVector(1)(image_dense)

        image_input = image_model.input
        return image_input, image_embedding

    def _build_word_embedding(self, vocabs):
        sentence_input = Input(shape=[None])
        if self._word_vector_init is None:
            word_embedding = Embedding(
                                    input_dim=self._vocab_size,
                                    output_dim=self._embedding_size,
                                    embeddings_regularizer=self._regularizer
                             )(sentence_input)
        else:
            WordVector = get_word_vector_class(self._word_vector_init)
            word_vector = WordVector(vocabs, self._initializer)
            embedding_weights = word_vector.vectorize_words(vocabs)
            word_embedding = Embedding(
                                    input_dim=self._vocab_size,
                                    output_dim=self._embedding_size,
                                    embeddings_regularizer=self._regularizer,
                                    weights=[embedding_weights]
                             )(sentence_input)
        return sentence_input, word_embedding

    def _build_sequence_model(self, sequence_input):
        RNN = GRU if self._rnn_type == 'gru' else LSTM

        def rnn():
            rnn = RNN(units=self._rnn_output_size,
                      return_sequences=True,
                      dropout=self._dropout_rate,
                      recurrent_dropout=self._dropout_rate,
                      kernel_regularizer=self._regularizer,
                      kernel_initializer=self._initializer,
                      implementation=2)
            rnn = Bidirectional(rnn) if self._bidirectional_rnn else rnn
            return rnn

        input_ = sequence_input
        for _ in range(self._rnn_layers):
            input_ = BatchNormalization(axis=-1)(input_)
            rnn_out = rnn()(input_)
            input_ = rnn_out
        time_dist_dense = TimeDistributed(Dense(units=self._vocab_size))(rnn_out)

        return time_dist_dense
