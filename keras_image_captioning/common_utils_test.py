from datetime import timedelta
from . import common_utils


def test_parse_timedelta():
    timedelta_str = '1 day, 12:30:00'
    timedelta_obj = common_utils.parse_timedelta(timedelta_str)
    assert timedelta_obj == timedelta(days=1, hours=12, minutes=30)


def test_parse_timedelta_with_no_day():
    timedelta_str = '0:30:00'
    timedelta_obj = common_utils.parse_timedelta(timedelta_str)
    assert timedelta_obj == timedelta(minutes=30)


def test_parse_timedelta_with_only_day():
    timedelta_str = '1 day'
    timedelta_obj = common_utils.parse_timedelta(timedelta_str)
    assert timedelta_obj == timedelta(days=1)


def test_parse_timedelta_with_timedelta_instance():
    timedelta_str = timedelta(hours=2)
    timedelta_obj = common_utils.parse_timedelta(timedelta_str)
    assert timedelta_obj == timedelta(hours=2)


def test_flatten_list_2d():
    list_2d = [[1, 2], [3]]
    assert common_utils.flatten_list_2d(list_2d) == [1, 2, 3]
