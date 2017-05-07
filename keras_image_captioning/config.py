import yaml
import sys

from collections import namedtuple
from datetime import timedelta
from random import choice, randint, uniform

from .common_utils import parse_timedelta


_REDUCE_LR_PATIENCE = 2
_EARLY_STOPPING_PATIENCE = 4

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
''')


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
                      initializer='glorot_uniform')


class RandomConfigBuilder(ConfigBuilderBase):
    _BATCH_SIZE = lambda _: choice([16, 32, 64])
    _REDUCE_LR_FACTOR = lambda _: uniform(0.1, 0.9)
    _LEMMATIZE_CAPTION = lambda _: choice([True, False])
    _RARE_WORDS_HANDLING = lambda _: choice(['nothing', 'discard', 'change'])
    _WORDS_MIN_OCCUR = lambda _: randint(1, 5)
    _LEARNING_RATE = lambda _: 10**uniform(-5, -3)
    _EMBEDDING_SIZE = lambda _: 100 * randint(1, 5)
    _RNN_OUTPUT_SIZE = lambda _: 100 * randint(1, 5)
    _DROPOUT_RATE = lambda _: uniform(0.1, 0.9)
    _BIDIRECTIONAL_RNN = lambda _: choice([True, False])
    _RNN_TYPE = lambda _: choice(['lstm', 'gru'])
    _RNN_LAYERS = lambda _: randint(1, 2)
    _L1_REG = lambda _: 10**uniform(-5, 5)
    _L2_REG = lambda _: 10**uniform(-5, 5)
    _INITIALIZER = lambda _: choice(['glorot_normal', 'glorot_uniform',
                                     'he_normal', 'he_uniform'])

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
        config_dict = dict(
            batch_size=self._BATCH_SIZE(),
            reduce_lr_factor=self._REDUCE_LR_FACTOR(),
            reduce_lr_patience=_REDUCE_LR_PATIENCE,
            early_stopping_patience=_EARLY_STOPPING_PATIENCE,
            lemmatize_caption=self._LEMMATIZE_CAPTION(),
            rare_words_handling='nothing',
            words_min_occur=1,
            learning_rate=self._LEARNING_RATE(),
            vocab_size=None,
            embedding_size=self._EMBEDDING_SIZE(),
            rnn_output_size=self._RNN_OUTPUT_SIZE(),
            dropout_rate=self._DROPOUT_RATE(),
            bidirectional_rnn=self._BIDIRECTIONAL_RNN(),
            rnn_type=self._RNN_TYPE(),
            rnn_layers=self._RNN_LAYERS(),
            l1_reg=self._L1_REG(),
            l2_reg=self._L2_REG(),
            initializer=self._INITIALIZER())

        config_dict.update(self._fixed_config_keys)

        return Config(**config_dict)


class FileConfigBuilder(ConfigBuilderBase):
    def __init__(self, yaml_path):
        self._yaml_path = yaml_path

    def build_config(self):
        with open(self._yaml_path) as yaml_file:
            config_dict = yaml.load(yaml_file)

        config_dict['time_limit'] = parse_timedelta(config_dict['time_limit'])
        return Config(**config_dict)


_active_config = DefaultConfigBuilder().build_config()


def active_config(new_active_config=None):
    if new_active_config:
        global _active_config
        _active_config = new_active_config
    return _active_config


def init_vocab_size(vocab_size):
    if vocab_size is None:
        raise ValueError('vocab_size cannot be None!')
    if _active_config.vocab_size:
        raise RuntimeError('vocab_size has been initialized before!')

    global _active_config
    _active_config = _active_config._replace(vocab_size=vocab_size)


def write_to_file(config, yaml_path):
    with open(yaml_path, 'w') as f:
        config_dict = dict(config._asdict())
        time_limit = config_dict['time_limit']
        if time_limit:
            config_dict['time_limit'] = str(time_limit)
        yaml.dump(config_dict, f, default_flow_style=False)
