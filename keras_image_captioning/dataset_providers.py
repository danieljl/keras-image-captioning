import numpy as np

from copy import copy
from math import ceil
from operator import attrgetter

from .common_utils import flatten_list_2d
from .config import active_config
from .datasets import get_dataset_instance
from .preprocessors import CaptionPreprocessor, ImagePreprocessor


class DatasetProvider(object):
    """Acts as an adapter of `Dataset` for Keras' `fit_generator` method."""
    def __init__(self,
                 batch_size=None,
                 dataset=None,
                 image_preprocessor=None,
                 caption_preprocessor=None,
                 single_caption=False):
        """
        If an arg is None, it will get its value from config.active_config.
        """
        self._batch_size = batch_size or active_config().batch_size
        self._dataset = (dataset or
                         get_dataset_instance(single_caption=single_caption))
        self._image_preprocessor = image_preprocessor or ImagePreprocessor()
        self._caption_preprocessor = (caption_preprocessor or
                                      CaptionPreprocessor())
        self._single_caption = single_caption
        self._build()

    @property
    def vocabs(self):
        return self._caption_preprocessor.vocabs

    @property
    def vocab_size(self):
        return self._caption_preprocessor.vocab_size

    @property
    def training_steps(self):
        return int(ceil(1. * self._dataset.training_set_size /
                        self._batch_size))

    @property
    def validation_steps(self):
        return int(ceil(1. * self._dataset.validation_set_size /
                        self._batch_size))

    @property
    def test_steps(self):
        return int(ceil(1. * self._dataset.test_set_size /
                        self._batch_size))

    @property
    def training_results_dir(self):
        return self._dataset.training_results_dir

    @property
    def caption_preprocessor(self):
        return self._caption_preprocessor

    def training_set(self, include_datum=False):
        for batch in self._batch_generator(self._dataset.training_set,
                                           include_datum,
                                           random_transform=True):
            yield batch

    def validation_set(self, include_datum=False):
        for batch in self._batch_generator(self._dataset.validation_set,
                                           include_datum,
                                           random_transform=False):
            yield batch

    def test_set(self, include_datum=False):
        for batch in self._batch_generator(self._dataset.test_set,
                                           include_datum,
                                           random_transform=False):
            yield batch

    def _build(self):
        training_set = self._dataset.training_set
        if self._single_caption:
            training_captions = map(attrgetter('all_captions_txt'),
                                    training_set)
            training_captions = flatten_list_2d(training_captions)
        else:
            training_captions = map(attrgetter('caption_txt'), training_set)
        self._caption_preprocessor.fit_on_captions(training_captions)

    def _batch_generator(self, datum_list, include_datum=False,
                         random_transform=True):
        # TODO Make it thread-safe. Currently only suitable for workers=1 in
        # fit_generator.
        datum_list = copy(datum_list)
        while True:
            np.random.shuffle(datum_list)
            datum_batch = []
            for datum in datum_list:
                datum_batch.append(datum)
                if len(datum_batch) >= self._batch_size:
                    yield self._preprocess_batch(datum_batch, include_datum,
                                                 random_transform)
                    datum_batch = []
            if datum_batch:
                yield self._preprocess_batch(datum_batch, include_datum,
                                             random_transform)

    def _preprocess_batch(self, datum_batch, include_datum=False,
                          random_transform=True):
        imgs_path = map(attrgetter('img_path'), datum_batch)
        captions_txt = map(attrgetter('caption_txt'), datum_batch)

        img_batch = self._image_preprocessor.preprocess_images(imgs_path,
                                                            random_transform)
        caption_batch = self._caption_preprocessor.encode_captions(captions_txt)

        imgs_input = self._image_preprocessor.preprocess_batch(img_batch)
        captions = self._caption_preprocessor.preprocess_batch(caption_batch)

        captions_input, captions_output = captions
        X, y = [imgs_input, captions_input], captions_output

        if include_datum:
            return X, y, datum_batch
        else:
            return X, y
