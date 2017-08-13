from __future__ import print_function

import errno
import os
import sys
import yaml

from datetime import datetime


def _path_from_here(path):
    result = os.path.join(os.path.dirname(__file__), path)
    return os.path.normpath(result)


_VAR_ROOT_DIR = _path_from_here('../var/')
_RESULTS_ROOT_DIR = _path_from_here('../results/')


def path_from_var_dir(*paths):
    return os.path.join(_VAR_ROOT_DIR, *paths)


def path_from_results_dir(*paths):
    return os.path.join(_RESULTS_ROOT_DIR, *paths)


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


def write_yaml_file(obj, path):
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def logging(*args, **kwargs):
    utc_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(utc_now, '|', *args, **kwargs)
    sys.stdout.flush()
