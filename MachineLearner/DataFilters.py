from datetime import datetime
import common.constants as const


def default_dataFilter(data):
    return True


def followers_dataFilter(data, threshold):
    return data[const.USER_FOLLOWERS_COLUMN] > threshold


def date_dataFilter(data, threshold_date):
    date = datetime.strptime(data[const.DATE_COLUMN], const.databaseDateFormat).date()
    return date < threshold_date


def data_companyFilter(data, compant_name):
    return data[const.COMPANY_COLUMN] == compant_name


def keywords_dataFilter(data, keywords):
    dataKeyword = data[const.SEARCH_KEYWORD_COLUMN]
    for keyword in keywords:
        if dataKeyword == keyword:
            return True
    return False


def data_predictionFilter(data, predictionThreshold):
    return abs(data[const.PREDICTION_COLUMN]) < predictionThreshold
