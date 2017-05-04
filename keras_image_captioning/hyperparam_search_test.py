import pytest
import sh

from .config import active_config, FileConfigBuilder
from .hyperparam_search import (itertools, main, HyperparamSearch,
                                TrainingCommand)


class TestHyperparamSearch(object):
    def test_run(self, mocker):
        config_filepath = '/tmp/keras_image_captioning.does_not_exist'
        mocker.patch.object(TrainingCommand, 'config_filepath',
                            mocker.PropertyMock(return_value=config_filepath))
        mocker.patch.object(HyperparamSearch, 'num_gpus',
                            mocker.PropertyMock(return_value=4))
        mocker.patch.object(itertools, 'count', lambda: range(12))

        search = HyperparamSearch(training_label_prefix='hpsearch-test',
                                  dataset_name='flickr8k',
                                  epochs=1)
        search.run()


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
        training_command._config_filepath += 'does_not_exist'
        running_command = training_command.execute()
        with pytest.raises(sh.ErrorReturnCode_1):
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


def test_main():
    # TODO
    pass
    # main(training_label_prefix='hpsearch-test', time_limit='1 day')
    # assert False
