import pytest

from datetime import timedelta

from . import training
from .config import DefaultConfigBuilder


class TestTraining(object):
    def test___init__(self):
        config = DefaultConfigBuilder().build_config()

        config = config._replace(epochs=None, time_limit=None)
        with pytest.raises(ValueError):
            training.Training('training-test', conf=config)

        config = config._replace(epochs=2, time_limit=timedelta(minutes=2))
        with pytest.raises(ValueError):
            training.Training('training-test', conf=config)

        config = config._replace(epochs=2, time_limit=None)
        training.Training('training-test', conf=config)

        config = config._replace(epochs=None, time_limit=timedelta(minutes=2))
        training.Training('training-test', conf=config)

    def test_run(self, mocker):
        mocker.patch.object(training.DatasetProvider, 'training_steps',
                            mocker.PropertyMock(return_value=2))
        mocker.patch.object(training.DatasetProvider, 'validation_steps',
                            mocker.PropertyMock(return_value=2))

        config = DefaultConfigBuilder().build_config()
        config = config._replace(epochs=2, batch_size=2)
        train = training.Training('training-test', conf=config)
        train.run()
