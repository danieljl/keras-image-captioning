import yaml

from collections import namedtuple
from random import choice, randint, uniform


Config = namedtuple('Config', '''
    dataset_name
    epochs
    batch_size
    lemmatize_caption
    rare_words_handling
    words_min_occur
    learning_rate
    vocab_size
    embedding_size
    lstm_output_size
    dropout_rate
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
                      epochs=1,
                      batch_size=32,
                      lemmatize_caption=True,
                      rare_words_handling='nothing',
                      words_min_occur=1,
                      learning_rate=0.001,
                      vocab_size=None,
                      embedding_size=300,
                      lstm_output_size=256,
                      dropout_rate=0.3)


class RandomConfigBuilder(ConfigBuilderBase):
    _BATCH_SIZE = lambda: choice([16, 32, 64])
    _LEMMATIZE_CAPTION = lambda: choice([True, False])
    _RARE_WORDS_HANDLING = lambda: choice(['nothing', 'discard', 'change'])
    _WORDS_MIN_OCCUR = lambda: randint(1, 5)
    _LEARNING_RATE = lambda: 10**uniform(-4, -1)
    _VOCAB_SIZE = lambda: None
    _EMBEDDING_SIZE = lambda: randint(50, 500)
    _LSTM_OUTPUT_SIZE = lambda: randint(50, 500)
    _DROPOUT_RATE = lambda: uniform(0, 1)

    def __init__(self, fixed_config_keys):
        """
        Args
          fixed_config_keys: dataset_name and epochs must exist
        """
        if not ('dataset_name' in fixed_config_keys and
                'epochs' in fixed_config_keys):
            raise ValueError('fixed_config_keys must contain both dataset_name'
                             ' and epochs!')
        self._fixed_config_keys = fixed_config_keys

    def build_config(self):
        config_dict = dict(
            batch_size=self._BATCH_SIZE(),
            lemmatize_caption=self._LEMMATIZE_CAPTION(),
            rare_words_handling=self._RARE_WORDS_HANDLING(),
            words_min_occur=self._WORDS_MIN_OCCUR(),
            learning_rate=self._LEARNING_RATE(),
            vocab_size=self._VOCAB_SIZE(),
            embedding_size=self._EMBEDDING_SIZE(),
            lstm_output_size=self._LSTM_OUTPUT_SIZE(),
            dropout_rate=self._DROPOUT_RATE())

        config_dict.update(self._fixed_config_keys)

        return Config(**config_dict)


class FileConfigBuilder(ConfigBuilderBase):
    def __init__(self, yaml_path):
        self._yaml_path = yaml_path

    def build_config(self):
        with open(self._yaml_path) as yaml_file:
            config_dict = yaml.load(yaml_file)
        return Config(**config_dict)


_active_config = DefaultConfigBuilder().build_config()


def active_config(new_active_config=None):
    if new_active_config:
        global _active_config
        _active_config = new_active_config
    return _active_config


def init_vocab_size(vocab_size):
    if _active_config.vocab_size:
        raise RuntimeError('vocab_size has been initialized before!')
    global _active_config
    _active_config = _active_config._replace(vocab_size=vocab_size)


def write_to_file(config, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(dict(config._asdict()), f, default_flow_style=False)
