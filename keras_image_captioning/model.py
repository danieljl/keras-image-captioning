import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, Sequential
from keras.layers import (Dense, Dropout, Embedding, Input, LSTM, Merge,
                          RepeatVector, TimeDistributed)


def build_model(vocab_size,
                embedding_size,
                lstm_output_size,
                dropout_rate):

    image_model = InceptionV3(include_top=True, weights='imagenet')
    for layer in image_model.layers:
        layer.trainable = False
    # Discard the last Dense layer
    flatten_tensor = image_model.layers[-2].output

    dense_embed = Dense(embedding_size)(flatten_tensor)
    dense_embed = Dropout(dropout_rate)(dense_embed)
    # Add timestep dimension
    dense_embed_expanded = RepeatVector(1)(dense_embed)
    image_embedding_model = Model(input=image_model.input,
                                  output=dense_embed_expanded)

    sentence_input = Input(shape=[None])
    word_embedding = Embedding(input_dim=vocab_size,
                               output_dim=embedding_size)(sentence_input)
    word_embedding = Dropout(dropout_rate)(word_embedding)
    word_embedding_model = Model(input=sentence_input, output=word_embedding)

    lstm_input_seq = Merge([image_embedding_model, word_embedding_model],
                           mode='concat', concat_axis=1)

    model = Sequential()
    model.add(lstm_input_seq)
    model.add(LSTM(output_dim=lstm_output_size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=vocab_size)))

    model.compile(optimizer='adam',
                  loss=categorical_crossentropy_from_logits,
                  metrics=[categorical_accuracy_with_variable_timestep])
    return model


def categorical_crossentropy_from_logits(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep/word
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)
    avg = tf.reduce_mean(loss)
    # We only care the average of this tensor's elements anyway
    return tf.fill([shape[0]], avg)


def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep/word
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros as they represent dummy/padding words.
    is_zero_y_true = tf.equal(y_true, 0)
    is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
    y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
    y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                           tf.argmax(y_pred, axis=1)),
                                  dtype=tf.float32))
