import pytest
import sh

from .config import active_config, FileConfigBuilder
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


class TestHyperparamSearch(object):
    def test_run(self, mocker):
        if DRY_RUN:
            mocker.patch.object(TrainingCommand, 'config_filepath',
                                mocker.PropertyMock(
                                            return_value=NOT_EXISTING_PATH))

        mocker.patch.object(HyperparamSearch, 'num_gpus',
                            mocker.PropertyMock(return_value=NUM_GPUS))
        mocker.patch.object(itertools, 'count', lambda: range(NUM_SEARCHES))

        search = HyperparamSearch(training_label_prefix='hpsearch-test/search',
                                  dataset_name='flickr8k',
                                  epochs=EPOCHS)
        search.run()
        assert all(not x.process.is_alive()[0] for x in search.running_history)


class TestTrainingCommand(object):
    @pytest.fixture
    def training_command(self):
        return TrainingCommand(training_label='hpsearch-test/training-command',
                               config=active_config(),
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

    def test__init_config_filepath(self, training_command):
        training_command._init_config_filepath()
        config_builder = FileConfigBuilder(training_command._config_filepath)
        config = config_builder.build_config()
        assert config == active_config()

    def test__init_log_filepath(self, training_command):
        training_command._init_log_filepath()
        assert training_command._log_filepath.find(
                                        training_command.training_label) != -1


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
    main(training_label_prefix='hpsearch-test/main', dataset_name='flickr8k',
         epochs=EPOCHS)
