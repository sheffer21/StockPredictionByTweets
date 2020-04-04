from datetime import datetime
from common import constants as const


def getPostTimeFromTimeStamp(timeStamp):  # Return time in format: 'HH-MM-SS'
    datetimeObj = datetime.strptime(timeStamp, const.databaseDateFormat)
    return datetimeObj.time()


def getPostDateFromTimeStamp(timeStamp):  # Return date in format: 'YY-MM-DD'
    datetimeObj = datetime.strptime(timeStamp, const.databaseDateFormat)
    return datetimeObj.date()


class Post:

    def __init__(self, postId, text, timeStamp, followers, companies, postKeyword):
        self.__p_id = postId
        self.__p_text = text
        self.__p_timeStamp = timeStamp
        self.__p_postKeyword = postKeyword
        self.__p_companies = companies
        self.__p_followers = followers
        self.__p_date = getPostDateFromTimeStamp(timeStamp)
        self.__p_time = getPostTimeFromTimeStamp(timeStamp)
        self.__p_post_stocks_info = {}

    @property
    def id(self):
        return self.__p_id

    @property
    def companiesList(self):
        return self.__p_companies

    @property
    def postKeyword(self):
        return self.__p_postKeyword

    @property
    def timeStamp(self):
        return self.__p_timeStamp

    @property
    def text(self):
        return self.__p_text

    @property
    def followers(self):
        return self.__p_followers

    @property
    def date(self):
        return self.__p_date

    @property
    def time(self):
        return self.__p_time

    @property
    def stocksInfo(self):
        return self.__p_post_stocks_info

    @property
    def description(self):
        postDescription = f"Post id: {self.__p_id},\n" \
                          f"\t text: {self.__p_text},\n" \
                          f"\t date: {self.__p_timeStamp},\n" \
                          f"\t user followers: {self.__p_followers},\n"

        companiesDescription = "This post is associated with the following companies:\n"

        for company in self.__p_companies:
            companiesDescription = companiesDescription + "\t" + company.description + "\n"

        return postDescription + companiesDescription

    def addCompany(self, company):
        self.__p_companies.append(company)

    def addCompanies(self, companiesList):
        self.__p_companies.extend(companiesList)

    def addStockInfo(self, stockSymbol, stockInfo):
        self.__p_post_stocks_info[stockSymbol] = stockInfo

    def getStockInfo(self, stockSymbol):
        return self.__p_post_stocks_info[stockSymbol]
