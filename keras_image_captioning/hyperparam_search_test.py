import os
import pytest
import sh
import shutil

from .config import active_config, FileConfigBuilder
from .io_utils import path_from_var_dir
from .hyperparam_search import (itertools, main, HyperparamSearch,
                                TrainingCommand)

# Because we spawn other processes, we cannot mock anything in them. So, we
# deliberately set an invalid config path in order for the test runs fast.
# When it is needed to do the real test, set DRY_RUN to False and set
# DatasetProvider.training_steps and DatasetProvider.validation_steps to 1.
DRY_RUN = True
NOT_EXISTING_PATH = '/tmp/keras_image_captioning/this.does_not_exist'

NUM_GPUS = 2
NUM_SEARCHES = 6
EPOCHS = 2


@pytest.fixture(scope='module')
def clean_up_training_result_dir():
    result_dir = path_from_var_dir('flickr8k/training-results/test/hpsearch')
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)


@pytest.mark.usefixtures('clean_up_training_result_dir')
class TestHyperparamSearch(object):
    def test__init__(self, mocker):
        mocker.patch.object(HyperparamSearch, 'num_gpus',
                            mocker.PropertyMock(return_value=NUM_GPUS))
        search = HyperparamSearch(training_label_prefix='test/hpsearch/init1',
                                  dataset_name='flickr8k',
                                  epochs=EPOCHS)
        assert search.num_gpus == NUM_GPUS

    def test___init___with_num_gpus(self):
        search = HyperparamSearch(training_label_prefix='test/hpsearch/init2',
                                  dataset_name='flickr8k',
                                  epochs=EPOCHS,
                                  num_gpus=NUM_GPUS + 1)
        assert search.num_gpus == NUM_GPUS + 1

    def test_run(self, mocker):
        if DRY_RUN:
            mocker.patch.object(TrainingCommand, 'config_filepath',
                                mocker.PropertyMock(
                                            return_value=NOT_EXISTING_PATH))

        mocker.patch.object(HyperparamSearch, 'num_gpus',
                            mocker.PropertyMock(return_value=NUM_GPUS))
        mocker.patch.object(itertools, 'count', lambda: range(NUM_SEARCHES))

        search = HyperparamSearch(training_label_prefix='test/hpsearch/search',
                                  dataset_name='flickr8k',
                                  epochs=EPOCHS)
        search.run()
        assert all(not x[1].process.is_alive()[0]
                   for x in search.running_commands)


@pytest.mark.usefixtures('clean_up_training_result_dir')
class TestTrainingCommand(object):
    @pytest.fixture
    def config_used(self):
        config = active_config()
        config = config._replace(epochs=2, time_limit=None, batch_size=2)
        return config

    @pytest.fixture
    def training_command(self):
        return TrainingCommand(training_label='test/hpsearch/training-command',
                               config=self.config_used(),
                               gpu_index=0,
                               background=True)

    def test_execute(self, training_command):
        finished = []

        def done_callback(cmd, success, exit_code):
            finished.append(True)

        training_command._done_callback = done_callback

        if DRY_RUN:
            training_command._config_filepath = NOT_EXISTING_PATH
            running_command = training_command.execute()
            with pytest.raises(sh.ErrorReturnCode_1):
                running_command.wait()
        else:
            running_command = training_command.execute()
            running_command.wait()

        assert len(finished) == 1 and finished[0]

    def test__init_config_filepath(self, training_command, config_used):
        training_command._init_config_filepath()
        config_builder = FileConfigBuilder(training_command._config_filepath)
        config = config_builder.build_config()
        assert config == config_used

    def test__init_log_filepath(self, training_command):
        training_command._init_log_filepath()
        assert training_command._log_filepath.find(
                                        training_command.training_label) != -1


@pytest.mark.usefixtures('clean_up_training_result_dir')
def test_main(mocker):
    if DRY_RUN:
        mocker.patch.object(TrainingCommand, 'config_filepath',
                            mocker.PropertyMock(
                                            return_value=NOT_EXISTING_PATH))

    mocker.patch.object(HyperparamSearch, 'num_gpus',
                        mocker.PropertyMock(return_value=NUM_GPUS))
    mocker.patch.object(itertools, 'count', lambda: range(NUM_SEARCHES))

    # Inside main() if wait=False is passed to executor.shutdown(), the mock
    # will be broken. It works only for a few miliseconds, but then poof! The
    # object is restored to the original. If wait=True, the mock will work
    # perfectly.
    main(training_label_prefix='test/hpsearch/main', dataset_name='flickr8k',
         epochs=EPOCHS)
