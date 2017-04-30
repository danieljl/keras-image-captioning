import os

from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             TensorBoard)

from . import config
from . import io_utils
from .dataset_providers import DatasetProvider
from .models import ImageCaptioningModel


class Training(object):
    def __init__(self,
                 training_label,
                 config_builder=None,
                 early_stopping_patience=2,
                 max_q_size=10,
                 workers=1,
                 verbose=1):
        self._training_label = training_label
        self._config_builder = config_builder or config.DefaultConfigBuilder()
        self._max_q_size = max_q_size
        self._workers = workers
        self._verbose = verbose

    def run(self):
        self._prepare_config_and_dataset_provider()
        self._init_result_dir()
        self._init_callbacks()
        self._write_config()
        self._epochs = config.active_config().epochs
        self._model = ImageCaptioningModel()
        self._model.keras_model.fit_generator(
                generator=self._dataset_provider.training_set(),
                steps_per_epoch=self._dataset_provider.training_steps,
                epochs=self._epochs,
                validation_data=self._dataset_provider.validation_set(),
                validation_steps=self._dataset_provider.validation_steps,
                max_q_size=self._max_q_size,
                workers=self._workers,
                callbacks=self._callbacks,
                verbose=self._verbose)

    @property
    def result_dir(self):
        return self._result_dir

    def _prepare_config_and_dataset_provider(self):
        config.register_config_builder(self._config_builder)
        self._dataset_provider = DatasetProvider()
        config.init_vocab_size(self._dataset_provider.vocab_size)

    def _init_result_dir(self):
        self._result_dir = os.path.join(
                                self._dataset_provider.training_results_dir,
                                self._training_label)
        io_utils.mkdir_p(self._result_dir)

    def _init_callbacks(self):
        CSV_FILENAME = 'metrics_log.csv'
        self._csv_filepath = self._path_from_result_dir(CSV_FILENAME)
        csv_logger = CSVLogger(filename=self._csv_filepath, append=True)

        CHECKPOINT_FILENAME = 'model-checkpoint.hdf5'
        self._checkpoint_filepath = self._path_from_result_dir(
                                                        CHECKPOINT_FILENAME)
        model_checkpoint = ModelCheckpoint(filepath=self._checkpoint_filepath,
                                           monitor='val_loss',
                                           mode='min',
                                           save_best_only=True,
                                           period=1,
                                           verbose=self._verbose)

        tensorboard = TensorBoard(log_dir=self._result_dir,
                                  histogram_freq=1,
                                  write_graph=True)

        earling_stopping = EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=0,
                                         verbose=self._verbose)

        # TODO Add LearningRateScheduler and ReduceLROnPlateau
        # TODO Add custom callbacks: StopAfterTimedelta and StopWhenFileExists

        self._callbacks = [csv_logger, model_checkpoint, tensorboard,
                           earling_stopping]

    def _write_config(self):
        CONFIG_FILENAME = 'hyperparams_config.yaml'
        self._config_filepath = self._path_from_result_dir(CONFIG_FILENAME)
        config.write_to_file(self._config_filepath)

    def _path_from_result_dir(self, *paths):
        return os.path.join(self._result_dir, *paths)


def main():
    training = Training(training_label='training-test')
    training.run()


if __name__ == '__main__':
    main()
