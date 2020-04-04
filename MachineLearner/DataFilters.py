from datetime import datetime

import common.constants as const


def default_dataFilter(data):
    return True


def followers_dataFilter(data, threshold):
    return data[const.USER_FOLLOWERS_COLUMN] > threshold


def date_dataFilter(data, threshold_date):
    date = datetime.strptime(data[const.DATE_COLUMN], const.databaseDateFormat).date()
    value = date < threshold_date
    return value
