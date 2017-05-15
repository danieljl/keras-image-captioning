import pytest

from itertools import islice
from operator import attrgetter
from types import GeneratorType

from .dataset_providers import DatasetProvider


class TestDatasetProvider(object):
    @pytest.fixture
    def dataset_provider(self):
        return DatasetProvider(batch_size=4)

    def test_training_set(self, dataset_provider, mocker):
        mocker.patch.object(dataset_provider, '_batch_generator',
                            lambda _, __: range(5))
        generator = dataset_provider.training_set()
        assert isinstance(generator, GeneratorType)
        assert list(generator) == range(5)

    def test__batch_generator(self, dataset_provider, mocker):
        mocker.patch.object(dataset_provider, '_preprocess_batch',
                            lambda x, _: x)

        datum_list = range(10)
        generator = dataset_provider._batch_generator(datum_list)
        results = [next(generator) for _ in range(4)]
        assert [len(x) for x in results] == [4, 4, 2, 4]
        assert sorted(sum(results[:-1], [])) == datum_list

        datum_list = range(12)
        generator = dataset_provider._batch_generator(datum_list)
        assert isinstance(generator, GeneratorType)

        results = list(islice(generator, 4))
        assert [len(x) for x in results] == [4, 4, 4, 4]
        assert sorted(sum(results[:-1], [])) == datum_list

    def test__preprocess_batch(self, dataset_provider):
        batch_size = 8
        batch = dataset_provider._dataset.training_set[:batch_size]
        results = dataset_provider._preprocess_batch(batch)
        (imgs_input, captions_input), captions_output = results

        assert imgs_input.ndim == 4
        assert captions_input.ndim == 2
        assert captions_output.ndim == 3

        assert imgs_input.shape[0] == batch_size
        assert captions_input.shape[0] == batch_size
        assert captions_output.shape[0] == batch_size

        image_preprocessor = dataset_provider._image_preprocessor
        assert imgs_input.shape[1:3] == image_preprocessor.IMAGE_SIZE
        assert captions_input.shape[1] == captions_output.shape[1] - 1
        assert captions_output.shape[2] == dataset_provider.vocab_size

    def test__preprocess_batch_with_include_datum(self, dataset_provider):
        batch_size = 8
        batch = dataset_provider._dataset.training_set[:batch_size]
        results = dataset_provider._preprocess_batch(batch, include_datum=True)
        (imgs_input, captions_input), captions_output, datum_batch = results

        all_captions_txt = map(attrgetter('all_captions_txt'), datum_batch)
        assert len(all_captions_txt) == batch_size
        assert all(len(captions) == 5 for captions in all_captions_txt)
