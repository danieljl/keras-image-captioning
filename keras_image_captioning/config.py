import itertools
import yaml
import sys

from collections import OrderedDict, namedtuple
from datetime import timedelta
from random import choice, randint, uniform

from .common_utils import parse_timedelta
from .io_utils import write_yaml_file

# List of all available hyperparameters
Config = namedtuple('Config', '''
    dataset_name
    epochs
    time_limit
    batch_size

    reduce_lr_factor
    reduce_lr_patience
    early_stopping_patience

    lemmatize_caption
    rare_words_handling
    words_min_occur

    learning_rate
    vocab_size
    embedding_size
    rnn_output_size
    dropout_rate

    bidirectional_rnn
    rnn_type
    rnn_layers

    l1_reg
    l2_reg

    initializer
    word_vector_init

    image_augmentation
''')

BEST_CONFIGS = OrderedDict([
    ('hpsearch/12-finer/0013',
     Config(batch_size=32,
            bidirectional_rnn=False,
            dataset_name='flickr8k',
            dropout_rate=0.13077696459186092,
            early_stopping_patience=6,
            embedding_size=226,
            epochs=21,
            initializer='he_normal',
            l1_reg=8.831598074868035e-08,
            l2_reg=1.3722161194141783e-07,
            learning_rate=0.0007725907034140148,
            lemmatize_caption=True,
            word_vector_init=None,
            rare_words_handling='nothing',
            reduce_lr_factor=0.999999,
            reduce_lr_patience=9223372036854775807,
            rnn_layers=3,
            rnn_output_size=226,
            rnn_type='lstm',
            time_limit=None,
            vocab_size=5578,
            words_min_occur=1,
            image_augmentation=False))
])


class ConfigBuilderBase(object):
    def build_config(self):
        raise NotImplementedError


class StaticConfigBuilder(ConfigBuilderBase):
    def __init__(self, config):
        self._config = config

    def build_config(self):
        return self._config


class DefaultConfigBuilder(ConfigBuilderBase):
    def build_config(self):
        return Config(dataset_name='flickr8k',
                      epochs=None,
                      time_limit=timedelta(hours=10),
                      batch_size=32,
                      # As nearest as possible to 1.0, but must not be >= 1.0
                      reduce_lr_factor=1.0 - 1e-6,
                      reduce_lr_patience=sys.maxsize,
                      early_stopping_patience=sys.maxsize,
                      lemmatize_caption=True,
                      rare_words_handling='nothing',
                      words_min_occur=1,
                      learning_rate=0.001,
                      vocab_size=None,
                      embedding_size=300,
                      rnn_output_size=256,
                      dropout_rate=0.3,
                      bidirectional_rnn=False,
                      rnn_type='lstm',
                      rnn_layers=1,
                      l1_reg=0.0,
                      l2_reg=0.0,
                      initializer='glorot_uniform',
                      word_vector_init=None,
                      image_augmentation=False)


class VinyalsConfigBuilder(ConfigBuilderBase):
    def build_config(self):
        return Config(dataset_name='flickr8k',
                      epochs=None,
                      time_limit=timedelta(hours=24),
                      batch_size=32,
                      reduce_lr_factor=0.7,
                      reduce_lr_patience=4,
                      early_stopping_patience=sys.maxsize,
                      lemmatize_caption=True,
                      rare_words_handling='discard',
                      words_min_occur=5,
                      learning_rate=0.001,
                      vocab_size=None,
                      embedding_size=512,
                      rnn_output_size=512,
                      dropout_rate=0.3,
                      bidirectional_rnn=False,
                      rnn_type='lstm',
                      rnn_layers=1,
                      l1_reg=0.0,
                      l2_reg=0.0,
                      initializer='vinyals_uniform',
                      word_vector_init=None,
                      image_augmentation=False)


class RandomConfigBuilder(ConfigBuilderBase):
    def __init__(self, fixed_config_keys):
        """
        Args
          fixed_config_keys: dataset_name must exist;
                             epochs xor time_limit must exist
        """
        if 'dataset_name' not in fixed_config_keys:
            raise ValueError('fixed_config_keys must contain dataset_name!')
        if not (bool(fixed_config_keys.get('epochs')) ^
                bool(fixed_config_keys.get('time_limit'))):
            raise ValueError('fixed_config_keys must contain either epochs or '
                             'time_limit, but not both!')

        self._fixed_config_keys = fixed_config_keys
        self._fixed_config_keys.setdefault('epochs', None)
        self._fixed_config_keys.setdefault('time_limit', None)

    def build_config(self):
        network_size = self._embedding_size()
        config_dict = dict(
            batch_size=self._batch_size(),
            reduce_lr_factor=self._reduce_lr_factor(),
            reduce_lr_patience=self._reduce_lr_patience(),
            early_stopping_patience=self._early_stopping_patience(),
            lemmatize_caption=self._lemmatize_caption(),
            rare_words_handling=self._rare_words_handling(),
            words_min_occur=self._words_min_occur(),
            learning_rate=self._learning_rate(),
            vocab_size=None,
            embedding_size=network_size,
            rnn_output_size=network_size,
            dropout_rate=self._dropout_rate(),
            bidirectional_rnn=self._bidirectional_rnn(),
            rnn_type=self._rnn_type(),
            rnn_layers=self._rnn_layers(),
            l1_reg=self._l1_reg(),
            l2_reg=self._l2_reg(),
            initializer=self._initializer(),
            word_vector_init=self._word_vector_init(),
            image_augmentation=False)  # Don't finetune

        config_dict.update(self._fixed_config_keys)

        return Config(**config_dict)


class CoarseRandomConfigBuilder(RandomConfigBuilder):
    def __init__(self, fixed_config_keys):
        super(CoarseRandomConfigBuilder, self).__init__(fixed_config_keys)

        self._batch_size = lambda: 32
        self._reduce_lr_factor = lambda: 1.0 - 1e-6
        self._reduce_lr_patience = lambda: sys.maxsize
        self._early_stopping_patience = lambda: 4
        self._lemmatize_caption = lambda: True
        self._rare_words_handling = lambda: 'nothing'
        self._words_min_occur = lambda: 1
        self._bidirectional_rnn = lambda: False
        self._initializer = lambda: 'he_normal'
        self._word_vector_init = lambda: None

        self._learning_rate = lambda: 10 ** uniform(-6, -2)
        self._dropout_rate = lambda: uniform(0, 1)
        self._l1_reg = lambda: 10 ** uniform(-7, 0)
        self._l2_reg = lambda: 10 ** uniform(-7, 0)

        self._embedding_size = lambda: int(2 ** uniform(6, 9))  # [64, 512]
        self._rnn_output_size = lambda: int(2 ** uniform(6, 9))  # [64, 512]
        self._rnn_type = lambda: choice(['lstm', 'gru'])
        self._rnn_layers = lambda: randint(1, 5)


class FineRandomConfigBuilder(CoarseRandomConfigBuilder):
    def __init__(self, fixed_config_keys):
        super(FineRandomConfigBuilder, self).__init__(fixed_config_keys)

        self._early_stopping_patience = lambda: 6

        self._learning_rate = lambda: 10 ** uniform(-4, -2)
        self._dropout_rate = lambda: uniform(0, 0.75)
        self._l1_reg = lambda: 10 ** uniform(-8, -4)
        self._l2_reg = lambda: 10 ** uniform(-8, -4)

        self._embedding_size = lambda: int(2 ** uniform(7, 8))  # [128, 256]
        self._rnn_output_size = lambda: int(2 ** uniform(7, 8))  # [128, 256]
        self._rnn_type = lambda: 'lstm'
        self._rnn_layers = lambda: randint(1, 3)


class Embed300RandomConfigBuilder(RandomConfigBuilder):
    def __init__(self, fixed_config_keys):
        super(Embed300RandomConfigBuilder, self).__init__(fixed_config_keys)

        self._batch_size = lambda: 32
        self._reduce_lr_factor = lambda: 1.0 - 1e-6
        self._reduce_lr_patience = lambda: sys.maxsize
        self._early_stopping_patience = lambda: 8
        self._lemmatize_caption = lambda: True
        self._rare_words_handling = lambda: 'discard'
        self._words_min_occur = lambda: 5
        self._bidirectional_rnn = lambda: False
        self._initializer = lambda: 'glorot_uniform'
        self._word_vector_init = lambda: choice(['glove', 'fasttext'])

        self._l1_reg = lambda: 0.0
        self._l2_reg = lambda: 0.0

        self._embedding_size = lambda: 300
        self._rnn_output_size = lambda: 300
        self._rnn_type = lambda: 'lstm'
        self._rnn_layers = lambda: randint(1, 2)

        self._learning_rate = lambda: 10**uniform(-4, -2)
        self._dropout_rate = lambda: uniform(0.1, 0.6)


class Embed300FineRandomConfigBuilder(Embed300RandomConfigBuilder):
    def __init__(self, fixed_config_keys):
        super(Embed300FineRandomConfigBuilder, self).__init__(
                                                            fixed_config_keys)

        self._reduce_lr_factor = lambda: uniform(0.1, 0.9)
        self._reduce_lr_patience = lambda: 4
        self._lemmatize_caption = lambda: False

        # Values below are from an analysis of hpsearch/16
        self._word_vector_init = lambda: 'glove'
        self._rnn_layers = lambda: randint(2, 3)
        self._learning_rate = lambda: 10 ** uniform(-4, -2.7)
        self._dropout_rate = lambda: uniform(0.2, 0.5)


class FileConfigBuilder(ConfigBuilderBase):
    def __init__(self, yaml_path):
        self._yaml_path = yaml_path

    def build_config(self):
        with open(self._yaml_path) as yaml_file:
            config_dict = yaml.load(yaml_file)

        # For backward compatibility
        for field in Config._fields:
            config_dict.setdefault(field, None)

        config_dict['time_limit'] = parse_timedelta(config_dict['time_limit'])
        return Config(**config_dict)


_active_config = VinyalsConfigBuilder().build_config()


def active_config(new_active_config=None):
    if new_active_config:
        global _active_config
        _active_config = new_active_config
    return _active_config


def init_vocab_size(vocab_size):
    global _active_config
    _active_config = _active_config._replace(vocab_size=vocab_size)


def write_to_file(config, yaml_path):
    config_dict = dict(config._asdict())
    time_limit = config_dict['time_limit']
    if time_limit:
        config_dict['time_limit'] = str(time_limit)
    write_yaml_file(config_dict, yaml_path)
