import numpy as np
import pytest
import tensorflow as tf

from .losses import categorical_crossentropy_from_logits


@pytest.fixture
def session():
    return tf.Session()


def test_categorical_crossentropy_from_logits(session):
    batch, timestep, vocab_size = 8, 4, 2
    shape = (batch, timestep, vocab_size)
    # tf.random_normal is not used because every executing Session.run,
    # the value changes.
    y_true = tf.constant(np.random.normal(size=shape))
    y_pred = tf.constant(np.random.normal(size=shape))

    # Confirm the shape
    result = categorical_crossentropy_from_logits(y_true, y_pred)
    loss = session.run(result)
    assert loss.shape == (batch, timestep - 1)

    # Confirm that the last timestep/word (dummy) is discarded

    y_true_np = session.run(y_true)
    y_pred_np = session.run(y_pred)

    y_true_np_changed = y_true_np.copy()
    y_pred_np_changed = y_pred_np.copy()
    y_true_np_changed[:, -1, :] = np.random.normal(size=(batch, vocab_size))
    y_pred_np_changed[:, -1, :] = np.random.normal(size=(batch, vocab_size))

    y_true_changed = tf.constant(y_true_np_changed)
    y_pred_changed = tf.constant(y_pred_np_changed)

    result = categorical_crossentropy_from_logits(y_true_changed,
                                                  y_pred_changed)
    loss_changed = session.run(result)
    np.testing.assert_array_almost_equal(loss, loss_changed)

    # Confirm that padding words don't contribute to the loss

    # Mustn't be -1 (the last) since it is a dummy word that will be discarded
    dummy_index = -2

    y_true_np_dummy = y_true_np.copy()
    y_true_np_dummy[1, dummy_index, :] = np.zeros(shape=(vocab_size,))
    y_true_dummy = tf.constant(y_true_np_dummy)

    result = categorical_crossentropy_from_logits(y_true_dummy, y_pred)
    loss_before = session.run(result)

    y_pred_np_after = y_pred_np.copy()
    y_pred_np_after[1, dummy_index, :] = np.random.normal(size=(vocab_size,))
    y_pred_after = tf.constant(y_pred_np_after)

    result = categorical_crossentropy_from_logits(y_true_dummy, y_pred_after)
    loss_after = session.run(result)

    np.testing.assert_array_almost_equal(loss_before, loss_after)
