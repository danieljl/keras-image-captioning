import numpy as np
import pytest

from .datasets import get_dataset_instance
from .preprocessors import CaptionPreprocessor, ImagePreprocessor


class TestImagePreprocessor(object):
    # TODO Test with image_data_generator is not None

    @pytest.fixture
    def flickr8k(self):
        return get_dataset_instance()

    @pytest.fixture
    def img_prep(self):
        return ImagePreprocessor()

    def test_preprocess_images(self, img_prep, mocker):
        mocker.patch.object(ImagePreprocessor, '_preprocess_an_image', len)
        img_paths = ['/tmp/img1', '/tmp/img22']
        result = img_prep.preprocess_images(img_paths)
        assert result == [9, 10]

    def test_preprocess_batch(self, img_prep):
        img_list = [1, 2, 3]
        result = img_prep.preprocess_batch(img_list)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test__preprocess_an_image(self, img_prep, flickr8k):
        datum = flickr8k.training_set[8]
        img_array = img_prep._preprocess_an_image(datum.img_path)
        assert img_array.shape == ImagePreprocessor.IMAGE_SIZE + (3,)


class TestCaptionPreprocessor(object):
    @pytest.fixture
    def caption_prep(self):
        return CaptionPreprocessor(rare_words_handling='nothing',
                                   words_min_occur=1)

    def test_vocab_size(self, caption_prep):
        with pytest.raises(AttributeError):
            caption_prep.vocab_size

        captions = ['keras', 'tensorflow', 'deep learning']
        caption_prep.fit_on_captions(captions)
        assert caption_prep.vocab_size == 4 + 1  # plus <eos>

    def test_fit_on_captions(self, caption_prep):
        captions = ['Keras', 'TensorFlow', 'deep learning: keras']
        caption_prep.fit_on_captions(captions)

        eos = CaptionPreprocessor.EOS_TOKEN
        assert (set(['keras', 'tensorflow', 'deep', 'learning', eos]) ==
                set(caption_prep._tokenizer.word_index.keys()))

        # Index starts from 1
        assert (set([1, 2, 3, 4, 5]) ==
                set(caption_prep._tokenizer.word_index.values()))

    def test_encode_captions(self, caption_prep):
        captions = ['keras', 'tensorflow', 'keras theano']
        caption_prep.fit_on_captions(captions)

        results = caption_prep.encode_captions(captions)
        assert type(results[0]) == list
        assert type(results[0][0]) == int
        assert map(len, results) == [2, 2, 3]  # plus <eos>

        # The only unseen word is 'pytorch'. 'tensorflow' and 'TensorFlow'
        # should be considered the same word.
        unseen = ['keras pytorch tensorflow', 'keras TensorFlow']
        results = caption_prep.encode_captions(captions + unseen)
        assert results[-1] == results[-2]  # 'pytorch' skipped

    def test_decode_captions(self, caption_prep):
        pass  # TODO

    def test_preprocess_batch(self, caption_prep):
        # The index starts from 1 so sequences_to_matrix needs a numpy array
        # with a size of num_words = word_index + 1
        caption_prep._tokenizer.num_words = 3 + 1
        captions_label_encoded = [[2, 3, 1], [2, 1]]
        captions_input_expected = np.array([[1, 2, 0], [1, 0, 0]])
        captions_output_expected = np.array([[[0, 1, 0],
                                              [0, 0, 1],
                                              [1, 0, 0],
                                              [0, 0, 0]],
                                             [[0, 1, 0],
                                              [1, 0, 0],
                                              [0, 0, 0],
                                              [0, 0, 0]]])

        results = caption_prep.preprocess_batch(captions_label_encoded)
        captions_input, captions_output = results
        np.testing.assert_array_equal(captions_input, captions_input_expected)
        np.testing.assert_array_equal(captions_output, captions_output_expected)

    def test__add_eos(self, caption_prep):
        captions = ['keras', 'tensorflow pytorch']
        results = caption_prep._add_eos(captions)
        eos = CaptionPreprocessor.EOS_TOKEN
        assert all(x.endswith(' ' + eos) for x in results)
