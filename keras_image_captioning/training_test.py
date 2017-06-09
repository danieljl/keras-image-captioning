import os
import pytest
import shutil
import sys

from datetime import timedelta

from . import config
from . import training
from .io_utils import path_from_var_dir


TRAINING_LABEL = 'test/training'


@pytest.fixture(scope='module')
def clean_up_training_result_dir():
    result_dir = path_from_var_dir('flickr8k/training-results',
                                   TRAINING_LABEL)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)


@pytest.mark.usefixtures('clean_up_training_result_dir')
class TestTraining(object):
    def test___init__(self):
        conf = config.DefaultConfigBuilder().build_config()

        conf = conf._replace(epochs=None, time_limit=None)
        with pytest.raises(ValueError):
            training.Training(TRAINING_LABEL + '/init1', conf=conf)

        conf = conf._replace(epochs=2, time_limit=timedelta(minutes=2))
        with pytest.raises(ValueError):
            training.Training(TRAINING_LABEL + '/init2', conf=conf)

        conf = conf._replace(epochs=2, time_limit=None)
        training.Training(TRAINING_LABEL + '/init3', conf=conf)

        with pytest.raises(ValueError):  # Duplicate training label
            training.Training(TRAINING_LABEL + '/init3', conf=conf)

        conf = conf._replace(epochs=None, time_limit=timedelta(minutes=2))
        train = training.Training(TRAINING_LABEL + '/init4', conf=conf)
        assert train._epochs == sys.maxsize

    def test_run(self, mocker):
        mocker.patch.object(training.DatasetProvider, 'training_steps',
                            mocker.PropertyMock(return_value=2))
        mocker.patch.object(training.DatasetProvider, 'validation_steps',
                            mocker.PropertyMock(return_value=2))
        logging_mock = mocker.patch.object(training, 'logging')

        conf = config.DefaultConfigBuilder().build_config()
        conf = conf._replace(epochs=2, time_limit=None, batch_size=2)
        train = training.Training(TRAINING_LABEL + '/run', conf=conf)
        train.run()
        assert logging_mock.call_count == 2


class TestCheckpoint(object):
    pass  # TODO


@pytest.mark.usefixtures('clean_up_training_result_dir')
def test_main(mocker):
    mocker.patch.object(training.Training, 'run', lambda _: None)

    with pytest.raises(ValueError):
        training.main(training_label=TRAINING_LABEL + '/norun',
                      from_training_dir='notnone',
                      epochs=2, time_limit='03:00:00')

    yaml_path = '/tmp/keras_img_cap_conf_test_main.yaml'
    conf = config.DefaultConfigBuilder().build_config()
    conf = conf._replace(epochs=None, time_limit=timedelta(seconds=15),
                         batch_size=2)
    config.write_to_file(conf, yaml_path)

    train = training.main(training_label=TRAINING_LABEL + '/main0',
                          _unit_test=True)
    assert train._config == config.DefaultConfigBuilder().build_config()

    train = training.main(training_label=TRAINING_LABEL + '/main1',
                          config_file=yaml_path,
                          _unit_test=True)
    assert train._config == conf

    train = training.main(training_label=TRAINING_LABEL + '/main2',
                          config_file=yaml_path,
                          _unit_test=True,
                          embedding_size=123)
    assert train._config == conf._replace(embedding_size=123)

    train = training.main(training_label=TRAINING_LABEL + '/main3',
                          config_file=yaml_path,
                          _unit_test=True,
                          epochs=9)
    assert train._config == conf._replace(epochs=9, time_limit=None)

    train = training.main(training_label=TRAINING_LABEL + '/main4',
                          config_file=yaml_path,
                          _unit_test=True,
                          time_limit='02:00:00')
    assert train._config == conf._replace(time_limit=timedelta(hours=2),
                                          epochs=None)
