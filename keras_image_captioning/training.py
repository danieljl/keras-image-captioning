import fire
import os
import signal
import sys
import traceback

from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)

from . import config
from . import io_utils
from .callbacks import (LogLearningRate, LogMetrics, LogTimestamp,
                        StopAfterTimedelta, StopWhenValLossExploding)
from .common_utils import parse_timedelta
from .io_utils import logging
from .dataset_providers import DatasetProvider
from .models import ImageCaptioningModel


class Training(object):
    """A training instance that primarily handles training callbacks, such as
    logging and early stopping."""
    def __init__(self,
                 training_label,
                 conf=None,
                 model_weights_path=None,
                 min_delta=1e-4,
                 min_lr=1e-7,
                 log_metrics_period=4,
                 explode_ratio=0.25,
                 explode_patience=2,
                 max_q_size=10,
                 workers=1,
                 verbose=1):
        """
        Args
          conf: an instance of config.Config; its properties:
            epochs
            time_limit
            reduce_lr_factor
            reduce_lr_patience
            early_stopping_patience
        """
        self._training_label = training_label
        self._config = conf or config.active_config()
        self._epochs = self._config.epochs
        self._time_limit = self._config.time_limit
        self._reduce_lr_factor = self._config.reduce_lr_factor
        self._reduce_lr_patience = self._config.reduce_lr_patience
        self._early_stopping_patience = self._config.early_stopping_patience
        self._model_weights_path = model_weights_path
        self._min_delta = min_delta
        self._min_lr = min_lr
        self._log_metrics_period = log_metrics_period
        self._explode_ratio = explode_ratio
        self._explode_patience = explode_patience
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
        logging('Building model..')
        self._model.build(self._dataset_provider.vocabs)
        if self._model_weights_path:
            logging('Loading model weights from {}..'.format(
                                                    self._model_weights_path))
            self.keras_model.load_weights(self._model_weights_path)

        # self._model.build() is expensive so it increases the chance of a race
        # condition. Checking self._stop_training will minimize it (but it is
        # still possible).
        if self._stop_training:
            self._stop_training = False
            return

        logging('Training {} is starting..'.format(self._training_label))
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

        CONFIG_FILENAME = 'hyperparams-config.yaml'
        config_filepath = self._path_from_result_dir(CONFIG_FILENAME)
        if os.path.exists(config_filepath):
            raise ValueError('Training label {} exists!'.format(
                             self._training_label))

        io_utils.mkdir_p(self._result_dir)

    def _init_callbacks(self):
        log_lr = LogLearningRate()
        log_ts = LogTimestamp()
        log_metrics = LogMetrics(self._dataset_provider,
                                 period=self._log_metrics_period)

        CSV_FILENAME = 'metrics-log.csv'
        self._csv_filepath = self._path_from_result_dir(CSV_FILENAME)
        csv_logger = CSVLogger(filename=self._csv_filepath)

        CHECKPOINT_FILENAME = 'model-weights.hdf5'
        self._checkpoint_filepath = self._path_from_result_dir(
                                                        CHECKPOINT_FILENAME)
        model_checkpoint = ModelCheckpoint(filepath=self._checkpoint_filepath,
                                           monitor='val_cider',
                                           mode='max',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           period=1,
                                           verbose=self._verbose)

        tensorboard = TensorBoard(log_dir=self._result_dir,
                                  write_graph=False)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      mode='min',
                                      epsilon=self._min_delta,
                                      factor=self._reduce_lr_factor,
                                      patience=self._reduce_lr_patience,
                                      min_lr=self._min_lr,
                                      verbose=self._verbose)

        earling_stopping = EarlyStopping(monitor='val_loss',
                                         mode='min',
                                         min_delta=self._min_delta,
                                         patience=self._early_stopping_patience,
                                         verbose=self._verbose)

        stop_after = StopAfterTimedelta(timedelta=self._time_limit,
                                        verbose=self._verbose)

        stop_when = StopWhenValLossExploding(ratio=self._explode_ratio,
                                             patience=self._explode_patience,
                                             verbose=self._verbose)

        # TODO Add LearningRateScheduler. Is it still needed?

        self._callbacks = [log_lr,  # Must be before tensorboard
                           log_metrics,  # Must be before model_checkpoint and
                                         # tensorboard
                           model_checkpoint,
                           tensorboard,  # Must be before log_ts
                           log_ts,  # Must be before csv_logger
                           csv_logger,
                           reduce_lr,  # Must be after csv_logger
                           stop_when,  # Must be the third last
                           earling_stopping,  # Must be the second last
                           stop_after]  # Must be the last

    def _write_active_config(self):
        CONFIG_FILENAME = 'hyperparams-config.yaml'
        self._config_filepath = self._path_from_result_dir(CONFIG_FILENAME)
        config.write_to_file(config.active_config(), self._config_filepath)

    def _path_from_result_dir(self, *paths):
        return os.path.join(self._result_dir, *paths)


class Checkpoint(object):
    def __init__(self,
                 new_training_label,
                 training_dir,
                 load_model_weights,
                 log_metrics_period,
                 config_override):
        if 'epochs' in config_override and 'time_limit' in config_override:
            raise ValueError('epochs and time_limit cannot be both passed!')
        self._new_training_label = new_training_label
        self._training_dir = training_dir
        self._load_model_weights = load_model_weights
        self._log_metrics_period = log_metrics_period
        self._config_override = config_override

    @property
    def training(self):
        training_dir = self._training_dir
        hyperparam_path = os.path.join(training_dir, 'hyperparams-config.yaml')
        model_weights_path = os.path.join(training_dir, 'model-weights.hdf5')

        config_builder = config.FileConfigBuilder(hyperparam_path)
        config_dict = config_builder.build_config()._asdict()
        if self._config_override:
            config_dict.update(self._config_override)
            config_dict['time_limit'] = parse_timedelta(
                                                    config_dict['time_limit'])
            if 'epochs' in self._config_override:
                config_dict['time_limit'] = None
            elif 'time_limit' in self._config_override:
                config_dict['epochs'] = None

        conf = config.Config(**config_dict)
        model_weights_path = (model_weights_path if self._load_model_weights
                              else None)
        return Training(training_label=self._new_training_label,
                        conf=conf,
                        model_weights_path=model_weights_path,
                        log_metrics_period=self._log_metrics_period,
                        explode_patience=sys.maxsize)


def main(training_label,
         from_training_dir=None,
         load_model_weights=False,
         log_metrics_period=4,
         _unit_test=False,
         **config_override):
    if from_training_dir:
        checkpoint = Checkpoint(new_training_label=training_label,
                                training_dir=from_training_dir,
                                load_model_weights=load_model_weights,
                                log_metrics_period=log_metrics_period,
                                config_override=config_override)
        training = checkpoint.training
    else:
        training = Training(training_label=training_label,
                            log_metrics_period=log_metrics_period)

    def handler(signum, frame):
        logging('Stopping training..')
        logging('(Training will stop after the current epoch)')
        try:
            training.stop_training()
        except:
            traceback.print_exc(file=sys.stderr)
    signal.signal(signal.SIGINT, handler)

    training.run()

    if _unit_test:
        return training


if __name__ == '__main__':
    fire.Fire(main)
