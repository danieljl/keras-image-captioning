import os
import re

from collections import defaultdict, namedtuple

from . import io_utils
from .config import active_config


Datum = namedtuple('Datum', 'img_filename img_path '
                            'caption_txt all_captions_txt')


class Dataset(object):
    _DATASET_DIR_NAME = 'dataset'
    _TRAINING_RESULTS_DIR_NAME = 'training-results'

    def __init__(self, dataset_name, lemmatize_caption, single_caption):
        self._lemmatize_caption = lemmatize_caption
        self._single_caption = single_caption
        self._root_path = io_utils.path_from_var_dir(dataset_name)
        self._create_dirs()

    @property
    def training_set(self):
        return self._training_set

    @property
    def validation_set(self):
        return self._validation_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def training_set_size(self):
        return len(self._training_set)

    @property
    def validation_set_size(self):
        return len(self._validation_set)

    @property
    def test_set_size(self):
        return len(self._test_set)

    @property
    def dataset_dir(self):
        return os.path.join(self._root_path, self._DATASET_DIR_NAME)

    @property
    def training_results_dir(self):
        return os.path.join(self._root_path, self._TRAINING_RESULTS_DIR_NAME)

    def _create_dirs(self):
        io_utils.mkdir_p(self.dataset_dir)
        io_utils.mkdir_p(self.training_results_dir)


class Flickr8kDataset(Dataset):
    DATASET_NAME = 'flickr8k'

    _TEXT_DIRNAME = 'Flickr8k_text'
    _CAPTION_LEMMATIZED_FILENAME = 'Flickr8k.lemma.token.txt'
    _CAPTION_RAW_FILENAME = 'Flickr8k.token.txt'

    _IMG_DIRNAME = 'Flickr8k_Dataset'
    _IMG_TRAINING_FILENAME = 'Flickr_8k.trainImages.txt'
    _IMG_VALIDATION_FILENAME = 'Flickr_8k.devImages.txt'
    _IMG_TEST_FILENAME = 'Flickr_8k.testImages.txt'

    def __init__(self, lemmatize_caption, single_caption=False):
        super(Flickr8kDataset, self).__init__(self.DATASET_NAME,
                                              lemmatize_caption,
                                              single_caption)
        self._build()

    def _build(self):
        self._captions_of = self._build_captions()
        self._training_set = self._build_set(self._IMG_TRAINING_FILENAME)
        self._validation_set = self._build_set(self._IMG_VALIDATION_FILENAME)
        self._test_set = self._build_set(self._IMG_TEST_FILENAME)

    def _build_captions(self):
        if self._lemmatize_caption:
            caption_filename = self._CAPTION_LEMMATIZED_FILENAME
        else:
            caption_filename = self._CAPTION_RAW_FILENAME

        caption_path = os.path.join(self.dataset_dir, self._TEXT_DIRNAME,
                                    caption_filename)

        lines = io_utils.read_text_file(caption_path)
        lines_splitted = map(lambda x: re.split(r'#\d\t', x), lines)
        captions_of = defaultdict(list)
        for img_filename, caption_txt in lines_splitted:
            captions_of[img_filename].append(caption_txt)

        return dict(captions_of)

    def _build_set(self, img_set_filename):
        img_set_filepath = os.path.join(self.dataset_dir, self._TEXT_DIRNAME,
                                        img_set_filename)
        img_filenames = io_utils.read_text_file(img_set_filepath)
        dataset = []
        for img_filename in img_filenames:
            img_path = os.path.join(self.dataset_dir, self._IMG_DIRNAME,
                                    img_filename)
            all_captions_txt = self._captions_of[img_filename]
            for caption_txt in all_captions_txt:
                dataset.append(Datum(img_filename=img_filename,
                                     img_path=img_path,
                                     caption_txt=caption_txt,
                                     all_captions_txt=all_captions_txt))
                if self._single_caption:
                    break

        return dataset


def get_dataset_instance(dataset_name=None, lemmatize_caption=None,
                         single_caption=False):
    """
    If an arg is None, it will get its value from config.active_config.
    """
    dataset_name = dataset_name or active_config().dataset_name
    lemmatize_caption = lemmatize_caption or active_config().lemmatize_caption

    for dataset_class in [Flickr8kDataset]:
        if dataset_class.DATASET_NAME == dataset_name:
            return dataset_class(lemmatize_caption=lemmatize_caption,
                                 single_caption=single_caption)

    raise ValueError('Cannot find {} dataset!'.format(dataset_name))
