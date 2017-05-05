from . import io_utils


def test_path_from_var_dir():
    assert io_utils.path_from_var_dir('something').endswith('var/something')


def test_mkdir_p():
    io_utils.mkdir_p('/tmp/tmpdir')


def test_read_text_file():
    lines_generator = io_utils.read_text_file('/proc/cpuinfo')
    lines = list(lines_generator)
    assert len(lines) > 0


def test_print_flush():
    io_utils.print_flush('foo bar')


def test_logging():
    io_utils.logging('foo bar')
