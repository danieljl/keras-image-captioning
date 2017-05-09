import yaml
import sys

from collections import namedtuple
from datetime import timedelta
from random import choice, randint, uniform

from .common_utils import parse_timedelta


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
            batch_size=self._batch_size(),
            reduce_lr_factor=self._reduce_lr_factor(),
            reduce_lr_patience=self._reduce_lr_patience(),
            early_stopping_patience=self._early_stopping_patience(),
            lemmatize_caption=self._lemmatize_caption(),
            rare_words_handling=self._rare_words_handling(),
            words_min_occur=self._words_min_occur(),
            learning_rate=self._learning_rate(),
            vocab_size=None,
            embedding_size=self._embedding_size(),
            rnn_output_size=self._rnn_output_size(),
            dropout_rate=self._dropout_rate(),
            bidirectional_rnn=self._bidirectional_rnn(),
            rnn_type=self._rnn_type(),
            rnn_layers=self._rnn_layers(),
            l1_reg=self._l1_reg(),
            l2_reg=self._l2_reg(),
            initializer=self._initializer())

        config_dict.update(self._fixed_config_keys)

        return Config(**config_dict)


class CoarseRandomConfigBuilder(RandomConfigBuilder):
    def __init__(self, fixed_config_keys, overfit=False):
        super(CoarseRandomConfigBuilder, self).__init__(fixed_config_keys)

        self._batch_size = lambda: 32
        self._reduce_lr_factor = lambda: 0.2
        self._reduce_lr_patience = lambda: 2
        self._early_stopping_patience = lambda: 4
        self._lemmatize_caption = lambda: True
        self._rare_words_handling = lambda: 'nothing'
        self._words_min_occur = lambda: 1

        self._learning_rate = lambda: 10**uniform(-6, 1)
        self._embedding_size = lambda: 50 * randint(1, 10)
        self._rnn_output_size = lambda: 50 * randint(1, 10)
        self._dropout_rate = lambda: uniform(0, 1) if not overfit else 0.0
        self._bidirectional_rnn = lambda: choice([True, False])
        self._rnn_type = lambda: choice(['lstm', 'gru'])
        self._rnn_layers = lambda: randint(1, 3)
        self._l1_reg = lambda: 10**uniform(-5, 5) if not overfit else 0.0
        self._l2_reg = lambda: 10**uniform(-5, 5) if not overfit else 0.0
        self._initializer = lambda: 'he_normal'


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
