#!/usr/bin/env python

import json
import numpy as np
import random
import re

from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.text import Tokenizer
from keras.applications import inception_v3
from operator import itemgetter

import io_utils

CAPTION_TYPES = ['caption_raw', 'caption_lemmatized']
INCEPTION_V3_IMG_SIZE = (299, 299)


def main():
    dataset = generate_dataset()
    dataset_flattened = flatten_dataset(dataset)
    tokenizer = save_word_index(dataset_flattened['training'])
    print('word index generated')
    save_dataset(dataset_flattened, tokenizer)


def generate_dataset():
    img_filenames = {}
    for dataset_type, file_path in [('training', 'Flickr_8k.trainImages.txt'),
                                    ('validation', 'Flickr_8k.devImages.txt'),
                                    ('testing', 'Flickr_8k.testImages.txt')]:
        img_filenames[dataset_type] = io_utils.read_text_file(
                io_utils.dataset_text_path(file_path))

    img_captions = {}
    for caption_type, file_path in [('raw', 'Flickr8k.token.txt'),
                                    ('lemmatized', 'Flickr8k.lemma.token.txt')]:
        lines = io_utils.read_text_file(io_utils.dataset_text_path(file_path))
        lines_splitted = map(lambda x: re.split(r'#\d\t', x), lines)
        img_captions[caption_type] = defaultdict(list)
        for img_path, caption in lines_splitted:
            img_captions[caption_type][img_path].append(caption)

    dataset = defaultdict(list)
    for dataset_type in ['training', 'validation', 'testing']:
        for img_filename in img_filenames[dataset_type]:
            caption_raw = img_captions['raw'][img_filename]
            caption_lemmatized = img_captions['lemmatized'][img_filename]
            datum = dict(img_filename=img_filename,
                         caption_raw=caption_raw,
                         caption_lemmatized=caption_lemmatized)
            dataset[dataset_type].append(datum)

    return dataset


def flatten_dataset(dataset):
    dataset_flattened = defaultdict(list)
    for dataset_type, data in dataset.iteritems():
        for datum in data:
            for caption_raw, caption_lemmatized in zip(
                    datum['caption_raw'],
                    datum['caption_lemmatized']):
                dataset_flattened[dataset_type].append(
                        dict(img_filename=datum['img_filename'],
                        caption_raw=caption_raw + ' <eos>',
                        caption_lemmatized=caption_lemmatized + ' <eos>'))

    random.seed(42)  # For reproducibility
    for datum_list in dataset_flattened.itervalues():
        random.shuffle(datum_list)

    random.seed()  # Reseed again

    return dataset_flattened


def save_word_index(training):
    for caption_type in CAPTION_TYPES:
        captions = map(itemgetter(caption_type), training)
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(captions)

        filename = 'word_index-{}.json'.format(caption_type)
        with open(io_utils.dataset_generated_path(filename), 'w') as f:
            json.dump(tokenizer.word_index, f, sort_keys=True)

    return tokenizer


def save_dataset(dataset_all, tokenizer, batch_size=256):
    executor = ProcessPoolExecutor()
    futures = []
    for dataset_type in ['training', 'validation', 'testing']:
        dataset = dataset_all[dataset_type]
        for batch_i, start in enumerate(range(0, len(dataset), batch_size)):
            batch = dataset[start : start + batch_size]
            futures.append(executor.submit(save_caption, dataset_type, batch,
                                           batch_i, tokenizer))
            futures.append(executor.submit(save_image, dataset_type, batch,
                                           batch_i))

    for future in as_completed(futures):
        print('{} generated'.format(future.result()))

    executor.shutdown()


def save_caption(dataset_type, batch, batch_number, tokenizer):
    for caption_type in CAPTION_TYPES:
        captions_txt = map(itemgetter(caption_type), batch)
        captions_encoded = tokenizer.texts_to_sequences(captions_txt)

        batch_filename = '{}-{:0>3}-{}.npz'.format(dataset_type,
                                                   batch_number,
                                                   caption_type)
        batch_filepath = io_utils.dataset_generated_path(batch_filename)
        np.savez_compressed(batch_filepath, allow_pickle=False,
                            **pack_list(captions_encoded))

    return batch_filename


def save_image(dataset_type, batch, batch_number):
    img_paths = map(io_utils.dataset_img_path,
                    map(itemgetter('img_filename'), batch))
    imgs = map(load_img_to_array, img_paths)
    imgs = map(inception_v3.preprocess_input, imgs)

    batch_filename = '{}-{:0>3}-image.npz'.format(dataset_type, batch_number)
    batch_filepath = io_utils.dataset_generated_path(batch_filename)
    np.savez_compressed(batch_filepath, allow_pickle=False,
                        **pack_list(imgs))

    return batch_filename


def load_img_to_array(path, target_size=INCEPTION_V3_IMG_SIZE):
    return img_to_array(load_img(path, target_size=target_size))


def pack_list(a_list):
    return {str(i): x for i, x in enumerate(a_list)}


if __name__ == '__main__':
    main()
