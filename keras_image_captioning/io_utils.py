import errno
import os


def _path_from_here(path):
    return os.path.join(os.path.dirname(__file__), path)


_VAR_ROOT_DIR = _path_from_here('../var/')


def path_from_var_dir(*paths):
    return os.path.join(_VAR_ROOT_DIR, *paths)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_text_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()
