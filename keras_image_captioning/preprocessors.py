import numpy as np

from keras.applications import inception_v3
from keras.preprocessing import sequence as keras_seq
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from .config import active_config


class ImagePreprocessor(object):
    IMAGE_SIZE = (299, 299)

    def __init__(self, image_data_generator=None):
        self._image_data_generator = image_data_generator
        self._do_random_transform = image_data_generator is not None

    def preprocess_images(self, img_paths):
        return map(self._preprocess_an_image, img_paths)

    def preprocess_batch(self, img_list):
        return np.array(img_list)

    def _preprocess_an_image(self, img_path):
        img = load_img(img_path, target_size=self.IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = inception_v3.preprocess_input(img_array)
        if self._do_random_transform:
            img_array = self._image_data_generator.random_transform(img_array)

        return img_array


class CaptionPreprocessor(object):
    EOS_TOKEN = 'zeosz'

    def __init__(self, rare_words_handling=None, words_min_occur=None):
        """
        If an arg is None, it will get its value from config.active_config.
        Args:
          rare_words_handling: {'nothing'|'discard'|'change'}
          words_min_occur: words whose occurrences are less than this are
                           considered rare words
        """
        self._tokenizer = Tokenizer()
        self._rare_words_handling = (rare_words_handling or
                                     active_config().rare_words_handling)
        self._words_min_occur = (words_min_occur or
                                 active_config().words_min_occur)
        self._word_of = {}

    @property
    def EOS_TOKEN_LABEL_ENCODED(self):
        return self._tokenizer.word_index[self.EOS_TOKEN]

    @property
    def vocabs(self):
        word_index = self._tokenizer.word_index
        return sorted(word_index, key=word_index.get)  # Sort by word's index

    @property
    def vocab_size(self):
        return len(self._tokenizer.word_index)

    def fit_on_captions(self, captions_txt):
        captions_txt = self._handle_rare_words(captions_txt)
        captions_txt = self._add_eos(captions_txt)
        self._tokenizer.fit_on_texts(captions_txt)
        self._word_of = {i: w for w, i in self._tokenizer.word_index.items()}

    def encode_captions(self, captions_txt):
        captions_txt = self._add_eos(captions_txt)
        return self._tokenizer.texts_to_sequences(captions_txt)

    def decode_captions(self, captions_output, captions_output_expected=None):
        """
        Args
          captions_output: 3-d array returned by a model's prediction; it's the
            same as captions_output returned by preprocess_batch
        """
        captions = captions_output[:, :-1, :]  # Discard the last word (dummy)
        label_encoded = captions.argmax(axis=-1)
        num_batches, num_words = label_encoded.shape

        if captions_output_expected is not None:
            caption_lengths = self._caption_lengths(captions_output_expected)
        else:
            caption_lengths = [num_words] * num_batches

        captions_str = []
        for caption_i in range(num_batches):
            caption_str = []
            for word_i in range(caption_lengths[caption_i]):
                label = label_encoded[caption_i, word_i]
                label += 1  # Real label = label in model + 1
                caption_str.append(self._word_of[label])
            captions_str.append(' '.join(caption_str))

        return captions_str

    def normalize_captions(self, captions_txt):
        captions_txt = self._add_eos(captions_txt)
        word_sequences = map(text_to_word_sequence, captions_txt)
        result = map(' '.join, word_sequences)
        return result

    def preprocess_batch(self, captions_label_encoded):
        captions = keras_seq.pad_sequences(captions_label_encoded,
                                           padding='post')
        # Because the number of timesteps/words resulted by the model is
        # maxlen(captions) + 1 (because the first "word" is the image).
        captions_extended1 = keras_seq.pad_sequences(captions,
                                                maxlen=captions.shape[-1] + 1,
                                                padding='post')
        captions_one_hot = map(self._tokenizer.sequences_to_matrix,
                               np.expand_dims(captions_extended1, -1))
        captions_one_hot = np.array(captions_one_hot, dtype='int')

        # Decrease/shift word index by 1.
        # Shifting `captions_one_hot` makes the padding word
        # (index=0, encoded=[1, 0, ...]) encoded all zeros ([0, 0, ...]),
        # so its cross entropy loss will be zero.
        captions_decreased = captions.copy()
        captions_decreased[captions_decreased > 0] -= 1
        captions_one_hot_shifted = captions_one_hot[:, :, 1:]

        captions_input = captions_decreased
        captions_output = captions_one_hot_shifted
        return captions_input, captions_output

    def _handle_rare_words(self, captions):
        if self._rare_words_handling == 'nothing':
            return captions
        elif self._rare_words_handling == 'discard':
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(captions)
            new_captions = []
            for caption in captions:
                words = text_to_word_sequence(caption)
                new_words = [w for w in words
                             if tokenizer.word_counts.get(w, 0) >=
                             self._words_min_occur]
                new_captions.append(' '.join(new_words))
            return new_captions

        raise NotImplementedError('rare_words_handling={} is not implemented '
                                  'yet!'.format(self._rare_words_handling))

    def _add_eos(self, captions):
        return map(lambda x: x + ' ' + self.EOS_TOKEN, captions)

    def _caption_lengths(self, captions_output):
        one_hot_sum = captions_output.sum(axis=2)
        return (one_hot_sum != 0).sum(axis=1)
