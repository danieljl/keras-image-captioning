import numpy as np

from functools import partial
from keras.applications import inception_v3
from keras.preprocessing import sequence as keras_seq
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from .config import active_config


class ImagePreprocessor(object):
    """A Inception v3 image preprocessor. Implements an image augmentation
    as well."""
    IMAGE_SIZE = (299, 299)

    def __init__(self, image_augmentation=None):
        self._image_data_generator = ImageDataGenerator(rotation_range=40,
                                                        width_shift_range=0.2,
                                                        height_shift_range=0.2,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True,
                                                        fill_mode='nearest')
        if image_augmentation is None:
            self._image_augmentation_switch = active_config().image_augmentation
        else:
            self._image_augmentation_switch = image_augmentation

    def preprocess_images(self, img_paths, random_transform=True):
        return map(partial(self._preprocess_an_image,
                           random_transform=random_transform),
                   img_paths)

    def preprocess_batch(self, img_list):
        return np.array(img_list)

    def _preprocess_an_image(self, img_path, random_transform=True):
        img = load_img(img_path, target_size=self.IMAGE_SIZE)
        img_array = img_to_array(img)
        if self._image_augmentation_switch and random_transform:
            img_array = self._image_data_generator.random_transform(img_array)
        img_array = inception_v3.preprocess_input(img_array)

        return img_array


class CaptionPreprocessor(object):
    """Preprocesses captions before feeded into the network."""

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

    # TODO Test method below
    def decode_captions_from_list2d(self, captions_encoded):
        """
        Args
          captions_encoded: 1-based (Tokenizer's), NOT 0-based (model's)
        """
        captions_decoded = []
        for caption_encoded in captions_encoded:
            words_decoded = []
            for word_encoded in caption_encoded:
                # No need of incrementing word_encoded
                words_decoded.append(self._word_of[word_encoded])
            captions_decoded.append(' '.join(words_decoded))
        return captions_decoded

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
