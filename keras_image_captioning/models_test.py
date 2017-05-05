import keras
import pytest

from . import config
from .models import ImageCaptioningModel


class TestImageCaptioningModel(object):
    # TODO Add more comprehensive tests

    @pytest.fixture
    def model(self):
        return ImageCaptioningModel(learning_rate=0.01,
                                    vocab_size=50,
                                    embedding_size=10,
                                    rnn_output_size=5,
                                    dropout_rate=0.)

    def test_with_args_from_config(self, mocker):
        # So that the original config will be restored after this test finishes.
        # It needs to be done because we play with a global var. Ugh!
        mocker.patch.object(config, '_active_config', config._active_config)

        with pytest.raises(ValueError):
            ImageCaptioningModel()

        config.init_vocab_size(100)
        model = ImageCaptioningModel()
        with pytest.raises(AttributeError):
            model.keras_model

        model.build()
        assert isinstance(model.keras_model, keras.models.Model)

    def test_with_explicit_args(self, model):
        with pytest.raises(AttributeError):
            model.keras_model
        model.build()
        assert isinstance(model.keras_model, keras.models.Model)

    def test_arg_bidirectional_rnn(self, model):
        model._bidirectional_rnn = not model._bidirectional_rnn
        model.build()
        assert isinstance(model.keras_model, keras.models.Model)

    def test_arg_rnn_type(self, model):
        model._rnn_type = 'gru' if model._rnn_type == 'lstm' else 'lstm'
        model.build()
        assert isinstance(model.keras_model, keras.models.Model)
