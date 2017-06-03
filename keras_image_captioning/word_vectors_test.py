import pytest

from .preprocessors import CaptionPreprocessor
from .word_vectors import Glove, Fasttext


class WordVectorTestBase(object):
    _WORD_VECTOR = None

    @pytest.fixture
    def word_vector(self, mocker):
        mocker.patch.object(self._WORD_VECTOR, '_PRETRAINED_PATH',
                            self._WORD_VECTOR._PRETRAINED_PATH + '.sample')
        vocab_words = ['.', 'znotexistz', 'a', 'i']
        initializer = 'zeros'
        word_vector = self._WORD_VECTOR(vocab_words=vocab_words,
                                        initializer=initializer)
        return word_vector

    def test___init__(self, word_vector):
        EOS_TOKEN = CaptionPreprocessor.EOS_TOKEN
        word_vector_of = word_vector._word_vector_of
        assert len(word_vector_of) == 3  # Not including znotexistz
        assert '.' not in word_vector_of
        assert 'znotexistz' not in word_vector_of
        assert EOS_TOKEN in word_vector_of
        assert 'a' in word_vector_of
        assert 'i' in word_vector_of

    def test_vectorize_words(self, word_vector):
        EOS_TOKEN = CaptionPreprocessor.EOS_TOKEN
        vectors = word_vector.vectorize_words(['qnotexistq', 'znotexistz', EOS_TOKEN,
                                         'a'])
        assert not vectors[:2].any()  # Assert all zeros
        assert vectors[2:].all()  # Assert all non-zeros


class TestGlove(WordVectorTestBase):
    _WORD_VECTOR = Glove


class TestFasttext(WordVectorTestBase):
    _WORD_VECTOR = Fasttext
