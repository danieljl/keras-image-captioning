#!/usr/bin/env python

import sys

from keras.callbacks import ModelCheckpoint, TensorBoard
from math import ceil

import io_utils
import model


def main(epochs):
    epochs = int(epochs)
    caption_type = 'lemmatized'
    batch_size = 32
    embedding_size = 300
    lstm_output_size = 256
    dropout_rate = 0.3

    tokenizer = io_utils.build_tokenizer(caption_type)
    the_model = model.build_model(vocab_size=len(tokenizer.word_index),
                                  embedding_size=embedding_size,
                                  lstm_output_size=lstm_output_size,
                                  dropout_rate=dropout_rate)
    training = io_utils.dataset_reader('training', caption_type, batch_size,
                                       tokenizer)
    validation = io_utils.dataset_reader('validation', caption_type,
                                         batch_size, tokenizer)

    model_path = io_utils.var_path('checkpoint/'
                                   'model.{epoch:03d}-{val_loss:.2f}.hdf5')
    model_checkpoint = ModelCheckpoint(model_path,
                                       monitor='val_loss',
                                       mode='min',
                                       period=1)
    log_dir = io_utils.var_path('tensorboard')
    tensorboard = TensorBoard(log_dir=log_dir,
                              histogram_freq=1,
                              write_graph=True)
    callbacks = [model_checkpoint, tensorboard]

    steps_per_epoch = int(ceil(1. * io_utils.NUM_TRAINING_SAMPLES /
                               batch_size))
    validation_steps = int(ceil(1. * io_utils.NUM_VALIDATION_SAMPLES /
                                batch_size))
    the_model.fit_generator(training,
                            steps_per_epoch,
                            epochs=epochs,
                            validation_data=validation,
                            validation_steps=validation_steps,
                            max_q_size=10,
                            workers=1,
                            callbacks=callbacks)


if __name__ == '__main__':
    main(*sys.argv[1:])
