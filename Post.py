from datetime import datetime


def getPostTimeFromTimeStamp(timeStamp):  # Return time in format: 'HH-MM-SS'
    datetimeObj = datetime.strptime(timeStamp, '%a %b %d %H:%M:%S +0000 %Y')
    return datetimeObj.time()


def getPostDateFromTimeStamp(timeStamp):  # Return date in format: 'YY-MM-DD'
    datetimeObj = datetime.strptime(timeStamp, '%a %b %d %H:%M:%S +0000 %Y')
    return datetimeObj.date()


class Post:

    def __init__(self, postId, text, timeStamp, source, companies, url, verified):
        self.__p_id = postId
        self.__p_text = text
        self.__p_timeStamp = timeStamp
        self.__p_source = source
        self.__p_companies = companies
        self.__p_url = url
        self.__p_verified = verified
        self.__p_date = getPostDateFromTimeStamp(timeStamp)
        self.__p_time = getPostTimeFromTimeStamp(timeStamp)
        self.__p_post_stocks_info = {}
        self.__p_total_impact = 0

    @property
    def id(self):
        return self.__p_id

    @property
    def totalImpact(self):
        return self.__p_total_impact

    @property
    def companiesList(self):
        return self.__p_companies

    @property
    def timeStamp(self):
        return self.__p_timeStamp

    @property
    def text(self):
        return self.__p_text

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
        postDescription = "Post id: {},\n" \
                          "\t text: {},\n" \
                          "\t date: {},\n" \
                          "\t source: {},\n" \
                          "\t url: {},\n" \
                          "\t verified: {},\n" \
                          "\t total impact: {},\n" \
                          "".format(self.__p_id, self.__p_text, self.__p_timeStamp, self.__p_source,
                                    self.__p_url, self.__p_verified, self.__p_total_impact)

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

    def setImpact(self, total_impact):
        self.__p_total_impact = total_impact

    def getStockInfo(self, stockSymbol):
        return self.__p_post_stocks_info[stockSymbol]
