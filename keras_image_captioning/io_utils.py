import errno
import json
import numpy as np
import os

from glob import glob
from keras.preprocessing import sequence as keras_seq
from keras.preprocessing import text as keras_txt


DATASET_ROOT_DIR = '../../datasets/flickr8k/'
VAR_ROOT_DIR = '../../var/'
NUM_TRAINING_SAMPLES = 5 * 6000
NUM_VALIDATION_SAMPLES = 5 * 1000
NUM_TESTING_SAMPLES = 5 * 1000


def dataset_path(path=''):
    return DATASET_ROOT_DIR + path


def dataset_img_path(path=''):
    return dataset_path('Flickr8k_Dataset/' + path)


def dataset_text_path(path=''):
    return dataset_path('Flickr8k_text/' + path)


def dataset_generated_path(path=''):
    return dataset_path('generated/' + path)


def var_path(path=''):
    return VAR_ROOT_DIR + path


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_text_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()


def load_dataset(filename):
    if not filename.startswith(dataset_generated_path()):
        filepath = dataset_generated_path(filename)
    else:
        filepath = filename
    with np.load(filepath) as f:
        batch_size = len(f.files) - 1  # One file is 'allow_pickle'
        dataset = [f[str(i)] for i in range(batch_size)]
        return dataset


def build_tokenizer(caption_type):
    word_index_filename = 'word_index-caption_{}.json'.format(caption_type)
    with open(dataset_generated_path(word_index_filename)) as f:
        word_index = json.load(f)

    tokenizer = keras_txt.Tokenizer(filters='')
    tokenizer.word_index = word_index
    return tokenizer


def dataset_reader(dataset_type, caption_type, batch_size, tokenizer):
    image_pattern = '{}-*-image.npz'.format(dataset_type)
    caption_pattern = '{}-*-caption_{}.npz'.format(dataset_type, caption_type)
    image_filepaths = sorted(glob(dataset_generated_path(image_pattern)))
    caption_filepaths = sorted(glob(dataset_generated_path(caption_pattern)))
    pair_filepaths = zip(image_filepaths, caption_filepaths)

    image_queue, caption_queue = [], []
    is_odd_batch = True
    while True:
        np.random.shuffle(pair_filepaths)
        for image_filepath, caption_filepath in pair_filepaths:
            image_queue.extend(load_dataset(image_filepath))
            caption_queue.extend(load_dataset(caption_filepath))

            # Before shuffling, there should be two batches in the queue in
            # order to be more random.
            if is_odd_batch:
                is_odd_batch = False
                continue
            else:
                is_odd_batch = True

            pair_queue = zip(image_queue, caption_queue)
            np.random.shuffle(pair_queue)
            image_queue, caption_queue = map(list, zip(*pair_queue))

            while len(image_queue) >= batch_size:
                image_batch = [image_queue.pop()
                               for _ in range(batch_size)]
                caption_batch = [caption_queue.pop()
                                 for _ in range(batch_size)]
                yield preprocess_batch(image_batch, caption_batch, tokenizer)


def preprocess_batch(image_batch, caption_batch, tokenizer):
    images = np.array(image_batch)

    captions = keras_seq.pad_sequences(caption_batch, padding='post')
    # Because the number of timesteps/words resulted by the model is
    # maxlen(captions) + 1 (because the first "word" is the image).
    captions_plus1 = keras_seq.pad_sequences(captions,
                                             maxlen=captions.shape[-1] + 1,
                                             padding='post')
    captions_encoded = map(tokenizer.sequences_to_matrix,
                           np.expand_dims(captions_plus1, -1))
    captions_encoded = np.array(captions_encoded, dtype='int')

    # Decrease/shift word index by 1. Shifting `captions_encoded` makes the
    # padding word (index=0, encoded=[1, 0, ...]) encoded all zeros
    # ([0, 0, ...]), so its cross entropy loss will be zero.
    captions_decreased = captions.copy()
    captions_decreased[captions_decreased > 0] -= 1
    captions_encoded_shifted = captions_encoded[:, :, 1:]

    X, y = [images, captions_decreased], captions_encoded_shifted
    return X, y
