import pytest

from datetime import timedelta

from . import config


class TestStaticConfigBuilder(object):
    def test_build_config(self):
        conf = config.DefaultConfigBuilder().build_config()
        builder = config.StaticConfigBuilder(conf)
        assert builder.build_config() == conf


class TestDefaultConfigBuilder(object):
    def test_build_config(self):
        conf = config.DefaultConfigBuilder().build_config()
        assert conf.epochs is None
        assert conf.time_limit is not None
        assert conf.vocab_size is None


class TestCoarseRandomConfigBuilder(object):
    def test_build_config_with_no_dataset_name(self):
        fixed_config_keys = {}
        with pytest.raises(ValueError):
            config.CoarseRandomConfigBuilder(fixed_config_keys)

    def test_build_config_with_neither_epochs_nor_time_limit(self):
        fixed_config_keys = dict(dataset_name='flickr8k')
        with pytest.raises(ValueError):
            config.CoarseRandomConfigBuilder(fixed_config_keys)

    def test_build_config_with_both_epochs_and_time_limit(self):
        fixed_config_keys = dict(dataset_name='flickr8k', epochs=1,
                                 time_limit=timedelta(minutes=1))
        with pytest.raises(ValueError):
            config.CoarseRandomConfigBuilder(fixed_config_keys)

    def test_build_config_with_proper_args(self):
        fixed_config_keys = dict(dataset_name='flickr8k',
                                 time_limit=timedelta(minutes=1),
                                 embedding_size=64)
        builder = config.CoarseRandomConfigBuilder(fixed_config_keys)
        conf = builder.build_config()
        assert conf.embedding_size == 64

        fixed_config_keys = dict(dataset_name='flickr8k',
                                 time_limit=timedelta(minutes=1))
        builder = config.CoarseRandomConfigBuilder(fixed_config_keys)
        conf = builder.build_config()
        assert conf.embedding_size == conf.rnn_output_size


class TestFileConfigBuilder(object):
    def test_build_config(self):
        yaml_path = '/tmp/keras_img_cap_conf.yaml'
        conf = config.DefaultConfigBuilder().build_config()
        config.write_to_file(conf, yaml_path)
        builder = config.FileConfigBuilder(yaml_path)
        assert builder.build_config() == conf


def test_active_config():
    default_config = config.VinyalsConfigBuilder().build_config()
    assert config.active_config() == default_config

    fixed_config_keys = dict(dataset_name='flickr8k',
                             time_limit=timedelta(minutes=1),
                             embedding_size=64)
    builder = config.CoarseRandomConfigBuilder(fixed_config_keys)
    random_config = builder.build_config()
    config.active_config(random_config)
    assert config.active_config() == random_config

    # Restore to the default config
    config.active_config(default_config)
    assert config.active_config() == default_config


def test_init_vocab_size(mocker):
    # So that the original config will be restored after this test finishes.
    # It needs to be done because we play with a global var. Ugh!
    mocker.patch.object(config, '_active_config', config._active_config)

    config.init_vocab_size(10)
    assert config.active_config().vocab_size == 10


def test_write_to_file():
    yaml_path = '/tmp/keras_img_cap_conf_2.yaml'
    config.write_to_file(config.active_config(), yaml_path)
