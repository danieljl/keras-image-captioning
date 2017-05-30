from . import io_utils


def test_path_from_var_dir():
    path = io_utils.path_from_var_dir('something')
    assert path.endswith('var/something')
    assert path.find('..') == -1


def test_mkdir_p():
    io_utils.mkdir_p('/tmp/tmpdir')


def test_read_text_file():
    lines_generator = io_utils.read_text_file('/proc/cpuinfo')
    lines = list(lines_generator)
    assert len(lines) > 0


def test_write_yaml_file():
    obj = dict(a=1, b='c')
    path = '/tmp/keras_img_cap_some.yaml'
    io_utils.write_yaml_file(obj, path)


def test_print_flush():
    io_utils.print_flush('foo bar')


def test_logging():
    io_utils.logging('foo bar')
