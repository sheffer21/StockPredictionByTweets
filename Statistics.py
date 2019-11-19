import datetime


class Statistics:

    __st_programStartTime = datetime.time()
    __st_post_by_years = {}
    __st_posts_by_dates = {}
    __st_total_posts = 0
    __st_total_companies = 0

    def __init__(self, programStartTime):
        self.__st_programStartTime = programStartTime
        self.__st_posts_by_years = {}
        self.__st_posts_by_dates = {}

    def addPostByYear(self, year):
        if year in self.__st_posts_by_years:
            self.__st_posts_by_years[year] += 1
        else:
            self.__st_posts_by_years[year] = 1

    def addPostByFullDate(self, date):
        if date in self.__st_posts_by_dates:
            self.__st_posts_by_dates[date] += 1
        else:
            self.__st_posts_by_dates[date] = 1

    def addPost(self):
        self.__st_total_posts += 1

    def adCCompany(self):
        self.__st_total_companies += 1

    def getStatistics(self):
        statisticsStr = ""
        return statisticsStr
