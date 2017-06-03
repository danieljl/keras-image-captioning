import numpy as np
from keras import backend as K, initializers

from .io_utils import path_from_var_dir
from .preprocessors import CaptionPreprocessor


class WordVector(object):
    def __init__(self, vocab_words, initializer):
        self._vocab_words = set(vocab_words)
        self._word_vector_of = dict()
        self._initializer = initializers.get(initializer)

    @property
    def embedding_size(self):
        return self._embedding_size

    def vectorize_words(self, words):
        sess = K.get_session()
        vectors = []
        for word in words:
            init = self._initializer(shape=(self._embedding_size,))
            vector = self._word_vector_of.get(word, sess.run(init))
            vectors.append(vector)
        return np.array(vectors)

    def _load_pretrained_vectors(self, file_obj):
        EOS_TOKEN = CaptionPreprocessor.EOS_TOKEN
        for line in file_obj:
            tokens = line.split()
            word = tokens[0]
            if word == '.':
                self._word_vector_of[EOS_TOKEN] = np.asarray(tokens[1:],
                                                             dtype='float32')
            elif word in self._vocab_words:
                self._word_vector_of[word] = np.asarray(tokens[1:],
                                                        dtype='float32')
        assert EOS_TOKEN in self._word_vector_of


class Glove(WordVector):
    _PRETRAINED_PATH = 'glove/glove.42B.300d.txt'

    def __init__(self, vocab_words, initializer):
        super(Glove, self).__init__(vocab_words, initializer)
        self._embedding_size = 300
        self._load_pretrained_vectors()

    def _load_pretrained_vectors(self):
        with open(_word_vectors_path(self._PRETRAINED_PATH)) as f:
            super(Glove, self)._load_pretrained_vectors(f)


class Fasttext(WordVector):
    _PRETRAINED_PATH = 'fasttext/wiki.en.vec'

    def __init__(self, vocab_words, initializer):
        super(Fasttext, self).__init__(vocab_words, initializer)
        self._embedding_size = 300
        self._load_pretrained_vectors()

    def _load_pretrained_vectors(self):
        with open(_word_vectors_path(self._PRETRAINED_PATH)) as f:
            next(f)  # Skip the first line as it is a header
            super(Fasttext, self)._load_pretrained_vectors(f)


def _word_vectors_path(*paths):
    return path_from_var_dir('pretrained-word-vectors', *paths)
