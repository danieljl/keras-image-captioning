DATASET_ROOT_DIR = '../../datasets/flickr8k/'


def dataset_path(path=''):
    return DATASET_ROOT_DIR + path


def dataset_img_path(path=''):
    return dataset_path('Flickr8k_Dataset/' + path)


def dataset_text_path(path=''):
    return dataset_path('Flickr8k_text/' + path)


def dataset_generated_path(path=''):
    return dataset_path('generated/' + path)


def read_text_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()
