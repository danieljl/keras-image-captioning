import os
import signal

from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)

from . import config
from . import io_utils
from .callbacks import LogLearningRate, LogTimestamp, StopAfterTimedelta
from .dataset_providers import DatasetProvider
from .models import ImageCaptioningModel


class Training(object):
    def __init__(self,
                 training_label,
                 config_=None,
                 time_limit=None,
                 reduce_lr_factor=0.5,
                 reduce_lr_patience=2,
                 early_stopping_patience=3,
                 min_loss_delta=1e-4,
                 max_q_size=10,
                 workers=1,
                 verbose=1):
        """
        Args
          time_limit: an instance of datetime.timedelta; if None, the training
                      stops until epochs reached; if set, the training stops
                      after it
        """
        self._training_label = training_label
        self._config = config_ or config.DefaultConfigBuilder().build_config()
        self._time_limit = time_limit
        self._reduce_lr_factor = reduce_lr_factor
        self._reduce_lr_patience = reduce_lr_patience
        self._early_stopping_patience = early_stopping_patience
        self._min_loss_delta = min_loss_delta
        self._max_q_size = max_q_size
        self._workers = workers
        self._verbose = verbose

        self._activate_config_and_init_dataset_provider()
        self._init_result_dir()
        self._init_callbacks()
        self._epochs = config.active_config().epochs
        self._model = ImageCaptioningModel()
        self._write_active_config()

    def run(self):
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
    def keras_model(self):
        return self._model.keras_model

    @property
    def result_dir(self):
        return self._result_dir

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


def main():
    training = Training(training_label='training-test')

    def handler(signum, frame):
        print('Stopping training...')
        print('(Training will stop after the current epoch)')
        training.keras_model.stop_training = True
    signal.signal(signal.SIGINT, handler)

    training.run()


if __name__ == '__main__':
    main()
