import pytest

from operator import attrgetter

from . import datasets


class TestFlickr8kDataset(object):

    @pytest.fixture
    def flickr8k(self):
        return datasets.Flickr8kDataset(True)

    def test_training_set(self, flickr8k):
        assert flickr8k.training_set_size == 6000 * 5
        assert (flickr8k.training_set[0].img_path.
                endswith('2513260012_03d33305cf.jpg'))
        assert (flickr8k.training_set[0].caption_txt ==
                'A black dog be run after a white dog in the snow .')

    def test_training_set_differs_from_validation_set(self, flickr8k):
        training_imgs = map(attrgetter('img_path'), flickr8k.training_set)
        validation_imgs = map(attrgetter('img_path'), flickr8k.validation_set)

        # Filenames only
        training_imgs = [x.split('/')[-1] for x in training_imgs]
        validation_imgs = [x.split('/')[-1] for x in validation_imgs]

        assert not (set(training_imgs) & set(validation_imgs))

    def test_dataset_dir(self, flickr8k):
        path = flickr8k.dataset_dir
        assert path.startswith('/home')
        assert path.endswith('/var/flickr8k/dataset')

    def test_training_results_dir(self, flickr8k):
        path = flickr8k.training_results_dir
        assert path.startswith('/home')
        assert path.endswith('/var/flickr8k/training-results')

    def test__build_captions(self, flickr8k):
        captions_of = flickr8k._build_captions()
        assert len(captions_of) == 40460 // 5
        assert (captions_of['1358089136_976e3d2e30.jpg'][0] ==
                'A boy sand surf down a hill')

    def test__build_set(self, flickr8k):
        img_set_filename = flickr8k._IMG_TEST_FILENAME
        test_set = flickr8k._build_set(img_set_filename)
        assert len(test_set) == 1000 * 5
        assert test_set[0].img_filename == '3385593926_d3e9c21170.jpg'
        assert test_set[0].img_path.endswith('3385593926_d3e9c21170.jpg')
        assert (test_set[0].caption_txt ==
                'Dog be in the snow in front of a fence .')
        assert len(test_set[0].all_captions_txt) == 5
