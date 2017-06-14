import numpy as np
import pytest

from .inference import BasicInference, BeamSearchInference, Caption, NLargest
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


# TODO Test with beam_size > 1
class TestBeamSearchInference(object):
    @pytest.fixture
    def inference(self, mocker):
        dataset_provider = DatasetProvider()

        def predict_on_batch(X):
            imgs_input, captions_input = X
            batch_size, caption_length = captions_input.shape
            return np.random.randn(batch_size, caption_length + 1,
                                   dataset_provider.vocab_size)

        keras_model = mocker.MagicMock()
        keras_model.predict_on_batch = predict_on_batch

        return BeamSearchInference(keras_model, dataset_provider,
                                   beam_size=1, max_caption_length=3)

    def test__predict_batch(self, inference):
        batch_size = 2
        imgs_input = np.random.randn(batch_size)
        result = inference._predict_batch(X=[imgs_input, None], y=None)
        assert len(result) == batch_size
        assert all(len(caption.split()) == 3 for caption in result)


class TestBatchNLargest(object):
    pass  # TODO


# TODO Test add_many
class TestNLargest(object):
    def test_functionality(self):
        captions = NLargest(2)
        captions.add(Caption(score=3, sentence='a'))
        captions.add(Caption(score=2, sentence='b'))
        captions.add(Caption(score=1, sentence='c'))
        expected = [Caption(score=3, sentence='a'),
                    Caption(score=2, sentence='b')]
        assert captions.n_largest(sort=True) == expected


class TestCaption(object):
    def test_comparison(self):
        c1 = Caption(score=2, sentence='a')
        c2 = Caption(score=1, sentence='b')
        assert c1 > c2
