import numpy as np
import pytest

from .inference import BasicInference
from .dataset_providers import DatasetProvider


class TestBasicInference(object):
    @pytest.fixture
    def inference(self, mocker):
        BATCH_SIZE = 2
        STEPS_PER_EPOCH = 3

        mocker.patch.object(DatasetProvider, 'training_steps',
                            mocker.PropertyMock(return_value=STEPS_PER_EPOCH))
        dataset_provider = DatasetProvider(batch_size=BATCH_SIZE)

        def predict_on_batch(X):
            imgs_input, captions_input = X
            batch_size, caption_length = captions_input.shape
            return np.random.randn(batch_size, caption_length + 1,
                                   dataset_provider.vocab_size)
        keras_model = mocker.MagicMock()
        keras_model.predict_on_batch = predict_on_batch

        return BasicInference(keras_model, dataset_provider)

    def test_predict_training_set(self, inference):
        result = inference.predict_training_set(include_datum=False)
        assert len(result) == 2 * 3  # BATCH_SIZE * STEPS_PER_EPOCH
        assert all(isinstance(x, str) for x in result)

    def test_evaluate_training_set(self, inference):
        result = inference.evaluate_training_set()
        assert all(isinstance(x, float) for x in result.values())
