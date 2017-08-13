import fire
import itertools
import os
import signal
import sh
import sys
import traceback

from concurrent.futures import ThreadPoolExecutor
from random import uniform
from threading import Lock, Semaphore
from time import sleep
from tempfile import gettempdir, NamedTemporaryFile

from .config import (active_config, write_to_file,
                     Embed300FineRandomConfigBuilder)
from .common_utils import parse_timedelta
from .datasets import get_dataset_instance
from .io_utils import mkdir_p, logging


class HyperparamSearch(object):
    """Spawns and schedules training scripts."""
    def __init__(self,
                 training_label_prefix,
                 dataset_name=None,
                 epochs=None,
                 time_limit=None,
                 num_gpus=None):
        if not ((epochs is None) ^ (time_limit is None)):
            raise ValueError('epochs or time_limit must present, '
                             'but not both!')

        self._training_label_prefix = training_label_prefix
        self._dataset_name = dataset_name or active_config().dataset_name
        self._validate_training_label_prefix()

        self._epochs = epochs
        self._time_limit = time_limit
        fixed_config_keys = dict(dataset_name=self._dataset_name,
                                 epochs=self._epochs,
                                 time_limit=self._time_limit)
        self._config_builder = Embed300FineRandomConfigBuilder(
                                                            fixed_config_keys)

        try:
            self._num_gpus = len(sh.nvidia_smi('-L').split('\n')) - 1
        except sh.CommandNotFound:
            self._num_gpus = 1
        self._num_gpus = num_gpus or self._num_gpus

        # TODO ! Replace set with a thread-safe set
        self._available_gpus = set(range(self.num_gpus))
        self._semaphore = Semaphore(self.num_gpus)
        self._running_commands = []  # a list of (index, sh.RunningCommand)
        self._stop_search = False
        self._lock = Lock()

    @property
    def training_label_prefix(self):
        return self._training_label_prefix

    @property
    def num_gpus(self):
        return self._num_gpus

    @property
    def running_commands(self):
        return self._running_commands

    @property
    def lock(self):
        return self._lock

    def run(self):
        """Start the hyperparameter search."""
        for search_index in itertools.count():
            sleep(uniform(0.1, 1))
            self._semaphore.acquire()

            with self.lock:
                if self._stop_search:
                    break

                training_label = self.training_label(search_index)
                config = self._config_builder.build_config()
                gpu_index = self._available_gpus.pop()
                done_callback = self._create_done_callback(gpu_index)

                command = TrainingCommand(training_label=training_label,
                                          config=config,
                                          gpu_index=gpu_index,
                                          background=True,
                                          done_callback=done_callback)
                self.running_commands.append((search_index, command.execute()))
                logging('Running training {}..'.format(training_label))

                self._remove_finished_commands()

        self._wait_running_commands()

    def stop(self):
        """Stop the hyperparameter search."""
        self._stop_search = True

    def training_label(self, search_index):
        return '{}/{:04d}'.format(self.training_label_prefix, search_index)

    def _validate_training_label_prefix(self):
        dataset = get_dataset_instance(self._dataset_name)
        prefix_dir = os.path.join(dataset.training_results_dir,
                                  self._training_label_prefix)
        if os.path.exists(prefix_dir):
            raise ValueError('Training label prefix {} exists!'.format(
                             self._training_label_prefix))

    def _create_done_callback(self, gpu_index):
        def done_callback(cmd, success, exit_code):
            # NEVER write anything to stdout in done_callback
            # OR a deadlock will happen

            self._available_gpus.add(gpu_index)
            self._semaphore.release()
        return done_callback

    def _remove_finished_commands(self):
        running_commands = []
        for search_index, running_command in self.running_commands:
            if running_command.process.is_alive()[0]:
                running_commands.append((search_index, running_command))
            else:
                training_label = self.training_label(search_index)
                logging('Training {} has finished.'.format(training_label))
        self._running_commands = running_commands

    def _wait_running_commands(self):
        for search_index, running_command in self.running_commands:
            training_label = self.training_label(search_index)
            logging('Waiting {} to finish..'.format(training_label))
            try:
                running_command.wait()
            except sh.ErrorReturnCode as e:
                logging('{} returned a non-zero code!'.format(training_label))
            except:
                traceback.print_exc(file=sys.stderr)


class TrainingCommand(object):
    """Executes and manages a training script."""

    COMMAND = sh.python.bake('-m', 'keras_image_captioning.training')

    def __init__(self,
                 training_label,
                 config,
                 gpu_index,
                 background=False,
                 done_callback=None):
        self._training_label = training_label
        self._config = config
        self._gpu_index = gpu_index
        self._background = background
        if done_callback is not None:
            self._done_callback = done_callback
        else:
            self._done_callback = lambda cmd, success, exit_code: None
        self._init_config_filepath()
        self._init_log_filepath()

    @property
    def training_label(self):
        return self._training_label

    @property
    def config(self):
        return self._config

    @property
    def gpu_index(self):
        return self._gpu_index

    @property
    def config_filepath(self):
        return self._config_filepath

    def execute(self):
        """Execute the training."""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_index)
        return self.COMMAND(training_label=self.training_label,
                            config_file=self.config_filepath,
                            _env=env,
                            _out=self._log_filepath,
                            _err_to_out=True,
                            _bg=self._background,
                            _done=self._done_callback)

    def _init_config_filepath(self):
        tmp_dir = os.path.join(gettempdir(), 'keras_image_captioning')
        mkdir_p(tmp_dir)
        config_file = NamedTemporaryFile(suffix='.yaml', dir=tmp_dir,
                                         delete=False)
        config_file.close()
        self._config_filepath = config_file.name
        write_to_file(self._config, self._config_filepath)

    def _init_log_filepath(self):
        LOG_FILENAME = 'training-log.txt'
        dataset = get_dataset_instance(self._config.dataset_name,
                                       self._config.lemmatize_caption)
        result_dir = os.path.join(dataset.training_results_dir,
                                  self._training_label)
        mkdir_p(result_dir)
        self._log_filepath = os.path.join(result_dir, LOG_FILENAME)


def main(training_label_prefix,
         dataset_name=None,
         epochs=None,
         time_limit=None,
         num_gpus=None):
    epochs = int(epochs) if epochs else None
    time_limit = parse_timedelta(time_limit) if time_limit else None
    num_gpus = int(num_gpus) if num_gpus else None
    search = HyperparamSearch(training_label_prefix=training_label_prefix,
                              dataset_name=dataset_name,
                              epochs=epochs,
                              time_limit=time_limit,
                              num_gpus=num_gpus)

    def handler(signum, frame):
        logging('Stopping hyperparam search..')
        with search.lock:
            search.stop()
            for index, running_command in search.running_commands:
                try:
                    label = search.training_label(index)
                    logging('Sending SIGINT to {}..'.format(label))
                    running_command.signal(signal.SIGINT)
                except OSError:  # The process might have exited before
                    logging('{} might have terminated before.'.format(label))
                except:
                    traceback.print_exc(file=sys.stderr)
            logging('All training processes have been sent SIGINT.')
    signal.signal(signal.SIGINT, handler)

    # We need to execute search.run() in another thread in order for Semaphore
    # inside it doesn't block the signal handler. Otherwise, the signal handler
    # will be executed after any training process finishes the whole epoch.

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(search.run)
    # wait must be True in order for the mock works,
    # see the unit test for more details
    executor.shutdown(wait=True)


if __name__ == '__main__':
    fire.Fire(main)
