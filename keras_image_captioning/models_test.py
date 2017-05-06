import keras
import pytest

from keras.regularizers import l1_l2

from . import config
from .models import ImageCaptioningModel

_VOCAB_SIZE = 100


class TestImageCaptioningModel(object):
    # TODO Add more comprehensive tests

    @pytest.fixture
    def model(self):
        return ImageCaptioningModel(learning_rate=0.01,
                                    vocab_size=_VOCAB_SIZE,
                                    embedding_size=10,
                                    rnn_output_size=5,
                                    dropout_rate=0.)

    def test_with_args_from_config(self, mocker):
        # So that the original config will be restored after this test finishes.
        # It needs to be done because we play with a global var. Ugh!
        mocker.patch.object(config, '_active_config', config._active_config)

        with pytest.raises(ValueError):
            ImageCaptioningModel()

        config.init_vocab_size(_VOCAB_SIZE)
        model = ImageCaptioningModel()
        with pytest.raises(AttributeError):
            model.keras_model

        self._build_and_assert(model)

    def test_with_explicit_args(self, model):
        with pytest.raises(AttributeError):
            model.keras_model
        self._build_and_assert(model)

    def test_arg_bidirectional_rnn(self, model):
        model._bidirectional_rnn = not model._bidirectional_rnn
        self._build_and_assert(model)

    def test_arg_rnn_type(self, model):
        model._rnn_type = 'gru' if model._rnn_type == 'lstm' else 'lstm'
        self._build_and_assert(model)

    def test_arg_rnn_layers(self, model):
        model._rnn_layers = 2 if model._rnn_layers == 1 else 1
        self._build_and_assert(model)

    def test_arg_rnn_layers_and_rnn_type(self, model):
        model._rnn_type = 'gru' if model._rnn_type == 'lstm' else 'lstm'
        model._rnn_layers = 2 if model._rnn_layers == 1 else 1
        self._build_and_assert(model)

    def test_arg_l1_reg_and_l2_reg(self, model):
        model._regularizer = l1_l2(0.01, 0.01)
        self._build_and_assert(model)

    def _build_and_assert(self, model):
        model.build()

        assert isinstance(model.keras_model, keras.models.Model)

        input_shape = model.keras_model.input_shape
        assert len(input_shape[0]) == 4  # batch, height, width, channel
        assert len(input_shape[1]) == 2  # batch, caption_length

        output_shape = model.keras_model.output_shape
        assert len(output_shape) == 3  # batch, caption_length, vocab_size
        assert output_shape[-1] == _VOCAB_SIZE
