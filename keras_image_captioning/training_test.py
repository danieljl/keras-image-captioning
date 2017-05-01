from . import training
from .config import DefaultConfigBuilder


class TestTraining(object):
    def test_run(self, mocker):
        mocker.patch.object(training.DatasetProvider, 'training_steps',
                            mocker.PropertyMock(return_value=2))
        mocker.patch.object(training.DatasetProvider, 'validation_steps',
                            mocker.PropertyMock(return_value=2))

        config = DefaultConfigBuilder().build_config()
        config = config._replace(epochs=2, batch_size=2)
        train = training.Training('training-test', config_=config)
        train.run()
