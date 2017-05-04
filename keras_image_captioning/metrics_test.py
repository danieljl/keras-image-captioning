import numpy as np
import tensorflow as tf

from .metrics import categorical_accuracy_with_variable_timestep


def test_categorical_accuracy_with_variable_timestep():
    # Confirm 100% accuracy when y_true and y_pred are exactly the same

    # Label-encoded: [[2, 3, 1], [2, 1]] -> [[1, 2, 0], [1, 0, 0]]
    # The additional [0, 0, 0] at the last for each batch is the dummy word
    y_true_np = np.array([[[0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 0],
                           [0, 0, 0]],
                          [[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]])
    y_pred_np = y_true_np.copy()
    _assert_accuracy(1.0, y_true_np, y_pred_np)

    # Confirm the accuracy value
    y_true_np = y_true_np.copy()
    y_pred_np = y_true_np.copy()
    y_pred_np[0, 0] = np.array([1, 0, 0])  # from [0, 1, 0]
    y_pred_np[1, 1] = np.array([0, 0, 1])  # from [1, 0, 0]
    _assert_accuracy(0.6, y_true_np, y_pred_np)

    # Confirm that last timestep/word (dummy) is discarded
    y_true_np = y_true_np.copy()
    y_pred_np = y_true_np.copy()
    y_true_np[:, -1, :] = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred_np[:, -1, :] = np.array([[0, 0, 1], [1, 0, 0]])
    _assert_accuracy(1.0, y_true_np, y_pred_np)

    # Confirm that the padding words don't contribute to the accuracy
    y_true_np = y_true_np.copy()
    y_pred_np = y_true_np.copy()
    y_pred_np[1, 2] = np.array([1, 0, 0])  # from [0, 0, 0]
    _assert_accuracy(1.0, y_true_np, y_pred_np)


def _assert_accuracy(accuracy_expected, y_true_np, y_pred_np):
    with tf.Session() as session:
        y_true = tf.constant(y_true_np)
        y_pred = tf.constant(y_pred_np)

        result = categorical_accuracy_with_variable_timestep(y_true, y_pred)
        accuracy = session.run(result)
        np.testing.assert_almost_equal(accuracy, accuracy_expected)
