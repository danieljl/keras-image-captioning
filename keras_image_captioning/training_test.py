import pytest
import sys

from datetime import timedelta

from . import config
from . import training


TRAINING_LABEL = 'test/training'


class TestTraining(object):
    def test___init__(self):
        conf = config.DefaultConfigBuilder().build_config()

        conf = conf._replace(epochs=None, time_limit=None)
        with pytest.raises(ValueError):
            training.Training(TRAINING_LABEL, conf=conf)

        conf = conf._replace(epochs=2, time_limit=timedelta(minutes=2))
        with pytest.raises(ValueError):
            training.Training(TRAINING_LABEL, conf=conf)

        conf = conf._replace(epochs=2, time_limit=None)
        training.Training(TRAINING_LABEL, conf=conf)

        conf = conf._replace(epochs=None, time_limit=timedelta(minutes=2))
        train = training.Training(TRAINING_LABEL, conf=conf)
        assert train._epochs == sys.maxsize

    def test_run(self, mocker):
        mocker.patch.object(training.DatasetProvider, 'training_steps',
                            mocker.PropertyMock(return_value=2))
        mocker.patch.object(training.DatasetProvider, 'validation_steps',
                            mocker.PropertyMock(return_value=2))

        conf = config.DefaultConfigBuilder().build_config()
        conf = conf._replace(epochs=2, batch_size=2)
        train = training.Training(TRAINING_LABEL, conf=conf)
        train.run()


def test_main(mocker):
    with pytest.raises(ValueError):
        training.main(training_label=TRAINING_LABEL, conf='foo')

    mocker.patch.object(training.Training, 'run', lambda _: None)

    yaml_path = '/tmp/keras_img_cap_conf_test_main.yaml'
    conf = config.DefaultConfigBuilder().build_config()
    conf = conf._replace(epochs=None, time_limit=timedelta(minutes=2))
    config.write_to_file(conf, yaml_path)

    train = training.main(training_label=TRAINING_LABEL,
                          config_file=yaml_path,
                          unit_test=True)
    assert train._config == conf
