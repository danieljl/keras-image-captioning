import re

from datetime import timedelta
from itertools import chain


def parse_timedelta(timedelta_str):
    if isinstance(timedelta_str, timedelta):
        return timedelta_str
    if not timedelta_str or timedelta_str == 'null':
        return None

    tokens = re.split(r' days?,? ', timedelta_str)
    if len(tokens) == 1:
        if tokens[0].find('day') == -1:
            days = '0'
            rest = tokens[0]
        else:
            days = tokens[0].split(' ')[0]
            rest = '00:00:00'
    else:
        days, rest = tokens

    days = int(days)
    hours, minutes, seconds = map(int, rest.split(':'))
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def flatten_list_2d(list_2d):
    return list(chain.from_iterable(list_2d))
