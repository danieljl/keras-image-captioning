import re
from datetime import timedelta


def parse_timedelta(timedelta_str):
    days, rest = re.split(r' days?, ', timedelta_str)
    days = int(days)
    hours, minutes, seconds = map(int, rest.split(':'))
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
