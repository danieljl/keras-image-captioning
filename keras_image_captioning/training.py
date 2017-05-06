import fire
import os
import signal
import sys
import traceback

from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)

from . import config
from . import io_utils
from .callbacks import LogLearningRate, LogTimestamp, StopAfterTimedelta
from .io_utils import logging
from .dataset_providers import DatasetProvider
from .models import ImageCaptioningModel


class Training(object):
    def __init__(self,
                 training_label,
                 conf=None,
                 reduce_lr_patience=2,
                 early_stopping_patience=4,
                 min_loss_delta=1e-4,
                 max_q_size=10,
                 workers=1,
                 verbose=1):
        """
        Args
          conf: an instance of config.Config; its properties:
            epochs
            time_limit
            reduce_lr_factor
        """
        self._training_label = training_label
        self._config = conf or config.DefaultConfigBuilder().build_config()
        self._epochs = self._config.epochs
        self._time_limit = self._config.time_limit
        self._reduce_lr_factor = self._config.reduce_lr_factor
        self._reduce_lr_patience = reduce_lr_patience
        self._early_stopping_patience = early_stopping_patience
        self._min_loss_delta = min_loss_delta
        self._max_q_size = max_q_size
        self._workers = workers
        self._verbose = verbose

        if not ((self._epochs is None) ^ (self._time_limit is None)):
            raise ValueError('Either conf.epochs or conf.time_limit must be '
                             'set, but not both!')

        if self._time_limit:
            self._epochs = sys.maxsize

        self._activate_config_and_init_dataset_provider()
        self._init_result_dir()
        self._init_callbacks()
        self._model = ImageCaptioningModel()
        self._write_active_config()

        self._stop_training = False

    @property
    def keras_model(self):
        return self._model.keras_model

    @property
    def result_dir(self):
        return self._result_dir

    def run(self):
        self._model.build()

        # self._model.build() is expensive so it increases the chance of a race
        # condition. Checking self._stop_training will minimize it (but it is
        # still possible).
        if self._stop_training:
            self._stop_training = False
            return

        self.keras_model.fit_generator(
                generator=self._dataset_provider.training_set(),
                steps_per_epoch=self._dataset_provider.training_steps,
                epochs=self._epochs,
                validation_data=self._dataset_provider.validation_set(),
                validation_steps=self._dataset_provider.validation_steps,
                max_q_size=self._max_q_size,
                workers=self._workers,
                callbacks=self._callbacks,
                verbose=self._verbose)

        self._stop_training = False
        logging('Training {} has finished.'.format(self._training_label))

    def stop_training(self):
        self._stop_training = True
        try:
            self.keras_model.stop_training = True
        # Race condition: ImageCaptioningModel.build is not called yet
        except AttributeError:
            pass

    def _activate_config_and_init_dataset_provider(self):
        config.active_config(self._config)
        self._dataset_provider = DatasetProvider()
        config.init_vocab_size(self._dataset_provider.vocab_size)

    def _init_result_dir(self):
        self._result_dir = os.path.join(
                                self._dataset_provider.training_results_dir,
                                self._training_label)
        io_utils.mkdir_p(self._result_dir)

    def _init_callbacks(self):
        log_lr = LogLearningRate()
        log_ts = LogTimestamp()

        CSV_FILENAME = 'metrics_log.csv'
        self._csv_filepath = self._path_from_result_dir(CSV_FILENAME)
        csv_logger = CSVLogger(filename=self._csv_filepath)

        CHECKPOINT_FILENAME = 'model-weights.hdf5'
        self._checkpoint_filepath = self._path_from_result_dir(
                                                        CHECKPOINT_FILENAME)
        model_checkpoint = ModelCheckpoint(filepath=self._checkpoint_filepath,
                                           monitor='val_loss',
                                           mode='min',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           period=1,
                                           verbose=self._verbose)

        tensorboard = TensorBoard(log_dir=self._result_dir,
                                  histogram_freq=1,
                                  write_graph=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      mode='min',
                                      epsilon=self._min_loss_delta,
                                      factor=self._reduce_lr_factor,
                                      patience=self._reduce_lr_patience,
                                      verbose=self._verbose)

        earling_stopping = EarlyStopping(monitor='val_loss',
                                         mode='min',
                                         min_delta=self._min_loss_delta,
                                         patience=self._early_stopping_patience,
                                         verbose=self._verbose)

        stop_after = StopAfterTimedelta(timedelta=self._time_limit,
                                        verbose=self._verbose)

        # TODO Add LearningRateScheduler. Is it still needed?

        self._callbacks = [log_lr,  # Must be before model_checkpoint
                           model_checkpoint,
                           tensorboard,  # Must be before log_ts
                           log_ts,  # Must be before csv_logger
                           csv_logger,
                           reduce_lr,  # Must be after csv_logger
                           earling_stopping,  # Must be the second last
                           stop_after]  # Must be the last

    def _write_active_config(self):
        CONFIG_FILENAME = 'hyperparams_config.yaml'
        self._config_filepath = self._path_from_result_dir(CONFIG_FILENAME)
        config.write_to_file(config.active_config(), self._config_filepath)

    def _path_from_result_dir(self, *paths):
        return os.path.join(self._result_dir, *paths)


def main(training_label, config_file=None, **kwargs):
    if 'conf' in kwargs:
        raise ValueError('conf must not be passed directly! '
                         'Use config_file instead.')

    unit_test = kwargs.pop('unit_test', False)

    if config_file is not None:
        config_builder = config.FileConfigBuilder(config_file)
        kwargs['conf'] = config_builder.build_config()

    training = Training(training_label, **kwargs)

    def handler(signum, frame):
        logging('Stopping training..')
        logging('(Training will stop after the current epoch)')
        try:
            training.stop_training()
        except:
            traceback.print_exc(file=sys.stderr)
    signal.signal(signal.SIGINT, handler)

    training.run()

    if unit_test:
        return training


if __name__ == '__main__':
    fire.Fire(main)
