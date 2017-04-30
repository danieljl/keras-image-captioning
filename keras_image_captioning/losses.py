import tensorflow as tf


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
