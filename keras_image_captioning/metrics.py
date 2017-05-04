import tensorflow as tf


def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    # Actually discarding is not needed if the dummy is an all-zeros array
    # (It is indeed encoded in an all-zeros array by
    # CaptionPreprocessing.preprocess_batch)
    y_true = y_true[:, :-1, :]  # Discard the last timestep/word (dummy)
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word (dummy)

    # Flatten the timestep dimension
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros as they represent dummy or padding words.
    is_zero_y_true = tf.equal(y_true, 0)
    is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
    y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
    y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                               tf.argmax(y_pred, axis=1)),
                                      dtype=tf.float32))
    return accuracy


# As Keras stores a function's name as its metric's name
categorical_accuracy_with_variable_timestep.__name__ = 'categorical_accuracy_wvt'
