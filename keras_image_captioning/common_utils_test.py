from datetime import timedelta
from . import common_utils


def test_parse_timedelta():
    timedelta_str = '1 day, 12:30:00'
    timedelta_obj = common_utils.parse_timedelta(timedelta_str)
    assert timedelta_obj == timedelta(days=1, hours=12, minutes=30)
