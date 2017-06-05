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
from .metrics import BLEU, CIDEr, ROUGE
from .models import ImageCaptioningModel


class BasicInference(object):
    _MAX_Q_SIZE = 10
    _WORKERS = 1
    _WAIT_TIME = 0.01

    def __init__(self, keras_model, dataset_provider):
        self._model = keras_model
        self._dataset_provider = dataset_provider
        self._preprocessor = dataset_provider.caption_preprocessor
        self._metrics = [BLEU(4), CIDEr(), ROUGE()]

    def predict_training_set(self, include_datum=True):
        return self._predict(self._dataset_provider.training_set,
                             self._dataset_provider.training_steps,
                             include_datum)

    def predict_validation_set(self, include_datum=True):
        return self._predict(self._dataset_provider.validation_set,
                             self._dataset_provider.validation_steps,
                             include_datum)

    def predict_testing_set(self, include_datum=True):
        return self._predict(self._dataset_provider.testing_set,
                             self._dataset_provider.testing_steps,
                             include_datum)

    def evaluate_training_set(self, include_prediction=False):
        return self._evaluate(self.predict_training_set(include_datum=True),
                              include_prediction=include_prediction)

    def evaluate_validation_set(self, include_prediction=False):
        return self._evaluate(self.predict_validation_set(include_datum=True),
                              include_prediction=include_prediction)

    def evaluate_testing_set(self, include_prediction=False):
        return self._evaluate(self.predict_testing_set(include_datum=True),
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
    def __init__(self,
                 keras_model,
                 beam_size=3,
                 max_caption_length=20):
        dataset_provider = DatasetProvider(single_caption=True)
        super(BeamSearchInference, self).__init__(keras_model,
                                                  dataset_provider)
        if beam_size > 1:
            raise NotImplementedError('Beam search with beam_size > 1 is not '
                                      'implemented yet!')
        self._beam_size = beam_size
        self._max_caption_length = max_caption_length

    def _predict_batch(self, X, y):
        imgs_input, _ = X
        batch_size = imgs_input.shape[0]

        captions_input = np.full((batch_size, 1),
                                 self._preprocessor.EOS_TOKEN_LABEL_ENCODED)
        captions_result = [None] * batch_size
        for _ in range(self._max_caption_length):
            captions_pred = self._model.predict_on_batch([imgs_input,
                                                          captions_input])
            captions_pred_str = self._preprocessor.decode_captions(
                                                captions_output=captions_pred)
            for i, caption in enumerate(captions_pred_str):
                if caption.endswith(' ' + self._preprocessor.EOS_TOKEN):
                    if captions_result[i] is None:
                        captions_result[i] = caption

            # If all reach <eos>
            if all(x is not None for x in captions_result):
                break

            encoded = self._preprocessor.encode_captions(captions_pred_str)
            captions_input, _ = self._preprocessor.preprocess_batch(encoded)

        # For captions that don't reach <eos> until the max caption length
        for i, caption in enumerate(captions_pred_str):
            if captions_result[i] is None:
                captions_result[i] = caption

        return captions_result


class NLargest(object):
    def __init__(self, n):
        self._n = n
        self._heap = []

    @property
    def size(self):
        return len(self._heap)

    def add(self, item):
        if len(self._heap) < self._n:
            heapq.heappush(self._heap, item)
        else:
            heapq.heappushpop(self._heap, item)

    def n_largest(self, sort=False):
        return sorted(self._heap, reverse=True) if sort else self._heap


# score should precede sentence so Caption is compared with score first
Caption = namedtuple('Caption', 'score sentence')


def main(training_dir, method='beam_search', beam_size=3):
    if method != 'beam_search':
        raise NotImplementedError('inference method = {} is not implemented '
                                  'yet!'.format(method))

    hyperparam_path = os.path.join(training_dir, 'hyperparams-config.yaml')
    model_weights_path = os.path.join(training_dir, 'model-weights.hdf5')

    logging('Loading hyperparams config..')
    config = FileConfigBuilder(hyperparam_path).build_config()
    active_config(config)
    model = ImageCaptioningModel()
    logging('Building model..')
    model.build()
    keras_model = model.keras_model
    logging('Loading model weights..')
    keras_model.load_weights(model_weights_path)

    inference = BeamSearchInference(keras_model, beam_size=beam_size)
    logging('Evaluating validation set..')
    metrics, predictions = inference.evaluate_validation_set(
                                                    include_prediction=True)

    logging('Writting result to files..')
    metrics_path = os.path.join(training_dir, 'validation-metrics.yaml')
    predictions_path = os.path.join(training_dir, 'validation-predictions.yaml')
    write_yaml_file(metrics, metrics_path)
    write_yaml_file(predictions, predictions_path)

    logging('Done!')


if __name__ == '__main__':
    fire.Fire(main)
