from keras.engine.training import GeneratorEnqueuer
from time import sleep

from .metrics import BLEU, CIDEr, ROUGE


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

    def evaluate_training_set(self):
        return self._evaluate(self.predict_training_set(include_datum=True))

    def evaluate_validation_set(self):
        return self._evaluate(self.predict_validation_set(include_datum=True))

    def evaluate_testing_set(self):
        return self._evaluate(self.predict_testing_set(include_datum=True))

    def _predict(self,
                 data_generator_function,
                 steps_per_epoch,
                 include_datum=True):
        data_generator = data_generator_function(include_datum=True)
        enqueuer = GeneratorEnqueuer(data_generator, pickle_safe=False)
        enqueuer.start(workers=self._WORKERS, max_q_size=self._MAX_Q_SIZE)

        caption_results = []
        datum_results = []
        for _ in range(steps_per_epoch):
            generator_output = None
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    sleep(self._WAIT_TIME)

            X, y, datum_batch = generator_output
            captions_pred = self._model.predict_on_batch(X)
            captions_pred_str = self._preprocessor.decode_captions(
                    captions_output=captions_pred,
                    captions_output_expected=y)
            caption_results += captions_pred_str
            datum_results += datum_batch

        enqueuer.stop()

        if include_datum:
            return zip(caption_results, datum_results)
        else:
            return caption_results

    def _evaluate(self, caption_datum_pairs):
        id_to_prediction = {}
        id_to_references = {}
        for caption_pred, datum in caption_datum_pairs:
            img_id = datum.img_filename
            caption_expected = self._preprocessor.normalize_captions(
                                                        [datum.caption_txt])
            id_to_prediction[img_id] = caption_pred
            id_to_references[img_id] = caption_expected

        result = {}
        for metric in self._metrics:
            metric_name_to_value = metric.calculate(id_to_prediction,
                                                    id_to_references)
            result.update(metric_name_to_value)
        return result
