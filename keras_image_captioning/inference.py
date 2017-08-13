import fire
import heapq
import numpy as np
import os

from collections import namedtuple
from keras.engine.training import GeneratorEnqueuer
from time import sleep
from tqdm import tqdm

from .config import FileConfigBuilder, active_config
from .dataset_providers import DatasetProvider
from .io_utils import logging, write_yaml_file
from .metrics import BLEU, CIDEr, METEOR, ROUGE
from .models import ImageCaptioningModel


class BasicInference(object):
    """A very basic inference without beam search. Technically, it is not an
    inference because the actual captions are also feeded into the model."""

    _MAX_Q_SIZE = 10
    _WORKERS = 1
    _WAIT_TIME = 0.01

    def __init__(self, keras_model, dataset_provider):
        self._model = keras_model
        self._dataset_provider = dataset_provider
        self._preprocessor = dataset_provider.caption_preprocessor
        self._metrics = [BLEU(4), METEOR(), CIDEr(), ROUGE()]

    def predict_training_set(self, include_datum=True):
        return self._predict(self._dataset_provider.training_set,
                             self._dataset_provider.training_steps,
                             include_datum)

    def predict_validation_set(self, include_datum=True):
        return self._predict(self._dataset_provider.validation_set,
                             self._dataset_provider.validation_steps,
                             include_datum)

    def predict_test_set(self, include_datum=True):
        return self._predict(self._dataset_provider.test_set,
                             self._dataset_provider.test_steps,
                             include_datum)

    def evaluate_training_set(self, include_prediction=False):
        return self._evaluate(self.predict_training_set(include_datum=True),
                              include_prediction=include_prediction)

    def evaluate_validation_set(self, include_prediction=False):
        return self._evaluate(self.predict_validation_set(include_datum=True),
                              include_prediction=include_prediction)

    def evaluate_test_set(self, include_prediction=False):
        return self._evaluate(self.predict_test_set(include_datum=True),
                              include_prediction=include_prediction)

    def _predict(self,
                 data_generator_function,
                 steps_per_epoch,
                 include_datum=True):
        data_generator = data_generator_function(include_datum=True)
        enqueuer = GeneratorEnqueuer(data_generator, pickle_safe=False)
        enqueuer.start(workers=self._WORKERS, max_q_size=self._MAX_Q_SIZE)

        caption_results = []
        datum_results = []
        for _ in tqdm(range(steps_per_epoch)):
            generator_output = None
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    sleep(self._WAIT_TIME)

            X, y, datum_batch = generator_output
            captions_pred_str = self._predict_batch(X, y)
            caption_results += captions_pred_str
            datum_results += datum_batch

        enqueuer.stop()

        if include_datum:
            return zip(caption_results, datum_results)
        else:
            return caption_results

    def _predict_batch(self, X, y):
        captions_pred = self._model.predict_on_batch(X)
        captions_pred_str = self._preprocessor.decode_captions(
                                                captions_output=captions_pred,
                                                captions_output_expected=y)
        return captions_pred_str

    def _evaluate(self, caption_datum_pairs, include_prediction=False):
        id_to_prediction = {}
        id_to_references = {}
        for caption_pred, datum in caption_datum_pairs:
            img_id = datum.img_filename
            caption_expected = self._preprocessor.normalize_captions(
                                                        datum.all_captions_txt)
            id_to_prediction[img_id] = caption_pred
            id_to_references[img_id] = caption_expected

        metrics = {}
        for metric in self._metrics:
            metric_name_to_value = metric.calculate(id_to_prediction,
                                                    id_to_references)
            metrics.update(metric_name_to_value)
        return (metrics, id_to_prediction) if include_prediction else metrics


class BeamSearchInference(BasicInference):
    """An implementation of inference using beam search."""
    def __init__(self,
                 keras_model,
                 beam_size=3,
                 max_caption_length=20):
        dataset_provider = DatasetProvider(single_caption=True)
        super(BeamSearchInference, self).__init__(keras_model,
                                                  dataset_provider)
        self._beam_size = beam_size
        self._max_caption_length = max_caption_length

    def _predict_batch(self, X, y):
        imgs_input, _ = X
        batch_size = imgs_input.shape[0]

        EOS_ENCODED = self._preprocessor.EOS_TOKEN_LABEL_ENCODED
        complete_captions = BatchNLargest(batch_size=batch_size,
                                          n=self._beam_size)
        partial_captions = BatchNLargest(batch_size=batch_size,
                                         n=self._beam_size)
        partial_captions.add([Caption(sentence_encoded=[EOS_ENCODED],
                                      log_prob=0.0)
                              for __ in range(batch_size)])

        for _ in range(self._max_caption_length):
            partial_captions_prev = partial_captions
            partial_captions = BatchNLargest(batch_size=batch_size,
                                             n=self._beam_size)

            for top_captions in partial_captions_prev.n_largest():
                sentences_encoded = [x.sentence_encoded for x in top_captions]
                captions_input, _ = self._preprocessor.preprocess_batch(
                                                            sentences_encoded)
                preds = self._model.predict_on_batch([imgs_input,
                                                      captions_input])
                preds = self._log_softmax(preds)  # Convert logits to log probs
                preds = preds[:, :-1, :]  # Discard the last word (dummy)
                preds = preds[:, -1]  # We only care the last word in a caption

                top_words = np.argpartition(
                                preds, -self._beam_size)[:, -self._beam_size:]
                row_indexes = np.arange(batch_size)[:, np.newaxis]
                top_words_log_prob = preds[row_indexes, top_words]
                log_probs_prev = np.array([x.log_prob
                                        for x in top_captions])[:, np.newaxis]
                log_probs_total = top_words_log_prob + log_probs_prev

                partial_captions_result = []
                complete_captions_result = []
                for sentence, words, log_probs in zip(sentences_encoded,
                                                      top_words,
                                                      log_probs_total):
                    partial_captions_batch = []
                    complete_captions_batch = []
                    for word, log_prob in zip(words, log_probs):
                        word += 1  # Convert from model's to Tokenizer's
                        # sentence[-1] is always EOS_ENCODED
                        sentence_encoded = sentence[:-1] + [word, sentence[-1]]
                        caption = Caption(sentence_encoded=sentence_encoded,
                                          log_prob=log_prob)
                        partial_captions_batch.append(caption)
                        if word == EOS_ENCODED:
                            complete_caption = Caption(
                                                    sentence_encoded=sentence,
                                                    log_prob=log_prob)
                            complete_captions_batch.append(complete_caption)
                        else:
                            complete_captions_batch.append(None)

                    partial_captions_result.append(partial_captions_batch)
                    complete_captions_result.append(complete_captions_batch)

                partial_captions.add_many(partial_captions_result)
                complete_captions.add_many(complete_captions_result)

        top_partial_captions = partial_captions.n_largest(sort=True)[0]
        top_complete_captions = complete_captions.n_largest(sort=True)[0]
        results = []
        for partial_caption, complete_caption in zip(top_partial_captions,
                                                     top_complete_captions):
            if complete_caption is None:
                results.append(partial_caption.sentence_encoded)
            else:
                results.append(complete_caption.sentence_encoded)

        return self._preprocessor.decode_captions_from_list2d(results)

    def _log_softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # For numerical stability
        return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


class BatchNLargest(object):
    """A batch priority queue."""

    def __init__(self, batch_size, n):
        self._batch_size = batch_size
        self._n_largests = [NLargest(n=n) for _ in range(batch_size)]

    def add(self, items):
        if len(items) != self._batch_size:
            raise ValueError('len of items must be equal to batch_size!')
        for n_largest, item in zip(self._n_largests, items):
            n_largest.add(item)

    def add_many(self, itemss):
        if len(itemss) != self._batch_size:
            raise ValueError('len of itemss must be equal to batch_size!')
        for n_largest, items in zip(self._n_largests, itemss):
            n_largest.add_many(items)

    def n_largest(self, sort=True):
        result = [x.n_largest(sort=sort) for x in self._n_largests]
        return map(list, map(None, *result))  # Transpose result


class NLargest(object):
    """An implementation of priority queue with max size."""

    def __init__(self, n):
        self._n = n
        self._heap = []

    def add(self, item):
        if item is None:
            return
        if len(self._heap) < self._n:
            heapq.heappush(self._heap, item)
        else:
            heapq.heappushpop(self._heap, item)

    def add_many(self, items):
        for item in items:
            self.add(item)

    def n_largest(self, sort=True):
        return sorted(self._heap, reverse=True) if sort else self._heap


# log_prob should precede sentence_encoded, so Caption is compared with
# log_prob first. sentence_encoded is 1-based (Tokenizer's), not 0-based
# (model's)
Caption = namedtuple('Caption', 'log_prob sentence_encoded')


def main(training_dir,
         dataset_type='validation',
         method='beam_search',
         beam_size=3,
         max_caption_length=20):
    if method != 'beam_search':
        raise NotImplementedError('inference method = {} is not implemented '
                                  'yet!'.format(method))
    if dataset_type not in ['validation', 'test']:
        raise ValueError('dataset_type={} is not recognized!'.format(
                                                                dataset_type))

    hyperparam_path = os.path.join(training_dir, 'hyperparams-config.yaml')
    model_weights_path = os.path.join(training_dir, 'model-weights.hdf5')

    logging('Loading hyperparams config..')
    config = FileConfigBuilder(hyperparam_path).build_config()
    config = config._replace(word_vector_init=None)  # As we do an inference
    active_config(config)
    model = ImageCaptioningModel()
    logging('Building model..')
    model.build()
    keras_model = model.keras_model
    logging('Loading model weights..')
    keras_model.load_weights(model_weights_path)

    inference = BeamSearchInference(keras_model,
                                    beam_size=beam_size,
                                    max_caption_length=max_caption_length)
    logging('Evaluating {} set..'.format(dataset_type))
    if dataset_type == 'test':
        metrics, predictions = inference.evaluate_test_set(
                                                    include_prediction=True)
    else:
        metrics, predictions = inference.evaluate_validation_set(
                                                    include_prediction=True)

    logging('Writting result to files..')
    metrics_path = os.path.join(training_dir,
            '{}-metrics-{}-{}.yaml'.format(dataset_type, beam_size,
                                           max_caption_length))
    predictions_path = os.path.join(training_dir,
            '{}-predictions-{}-{}.yaml'.format(dataset_type, beam_size,
                                               max_caption_length))
    write_yaml_file(metrics, metrics_path)
    write_yaml_file(predictions, predictions_path)

    logging('Done!')


if __name__ == '__main__':
    fire.Fire(main)
