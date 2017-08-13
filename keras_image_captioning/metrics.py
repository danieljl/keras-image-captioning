import os
import tensorflow as tf

from pycocoevalcap.bleu import bleu
from pycocoevalcap.cider import cider
from pycocoevalcap.meteor import meteor
from pycocoevalcap.rouge import rouge


class Score(object):
    """A subclass of this class is an adapter of pycocoevalcap."""

    def __init__(self, score_name, implementation):
        self._score_name = score_name
        self._implementation = implementation

    def calculate(self, id_to_prediction, id_to_references):
        id_to_preds = {}
        for id_, pred in id_to_prediction.items():
            id_to_preds[id_] = [pred]
        avg_score, scores = self._implementation.compute_score(
                                                id_to_references, id_to_preds)
        if isinstance(avg_score, (list, tuple)):
            avg_score = map(float, avg_score)
        else:
            avg_score = float(avg_score)
        return {self._score_name: avg_score}


class BLEU(Score):
    def __init__(self, n=4):
        implementation = bleu.Bleu(n)
        super(BLEU, self).__init__('bleu', implementation)
        self._n = n

    def calculate(self, id_to_prediction, id_to_references):
        name_to_score = super(BLEU, self).calculate(id_to_prediction,
                                                    id_to_references)
        scores = name_to_score.values()[0]
        result = {}
        for i, score in enumerate(scores, start=1):
            name = '{}_{}'.format(self._score_name, i)
            result[name] = score
        return result


class CIDEr(Score):
    def __init__(self):
        implementation = cider.Cider()
        super(CIDEr, self).__init__('cider', implementation)


class METEOR(Score):
    def __init__(self):
        implementation = meteor.Meteor()
        super(METEOR, self).__init__('meteor', implementation)

    def calculate(self, id_to_prediction, id_to_references):
        if self._data_downloaded():
            return super(METEOR, self).calculate(id_to_prediction,
                                                 id_to_references)
        else:
            return {self._score_name: 0.0}

    def _data_downloaded(self):
        meteor_dir = os.path.dirname(meteor.__file__)
        return (os.path.isfile(os.path.join(meteor_dir, 'meteor-1.5.jar')) and
                os.path.isfile(
                        os.path.join(meteor_dir, 'data', 'paraphrase-en.gz')))


class ROUGE(Score):
    def __init__(self):
        implementation = rouge.Rouge()
        super(ROUGE, self).__init__('rouge', implementation)


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
