import tensorflow as tf


def categorical_crossentropy_from_logits(y_true, y_pred):
    # Discarding is still needed although CaptionPreprocessor.preprocess_batch
    # has added dummy words as all-zeros arrays because the sum of losses is
    # the same but the mean of losses is different.
    y_true = y_true[:, :-1, :]  # Discard the last timestep/word (dummy)
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word (dummy)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)
    return loss
