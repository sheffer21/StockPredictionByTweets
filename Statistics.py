import datetime


class Statistics:

    __st_programStartTime = datetime.time()
    __st_posts_per_year = {}
    __st_posts_per_date = {}
    __st_totalPostsCount = 0
    __st_totalCompaniesCount = 0
    __st_successfulImportCount = 0
    __st_totalImportCount = 0

    def __init__(self, programStartTime):
        self.__st_programStartTime = programStartTime

    @property
    def totalPostsCount(self):
        return self.__st_totalPostsCount

    @property
    def totalCompaniesCount(self):
        return self.__st_totalCompaniesCount

    @property
    def totalImportCount(self):
        return self.__st_totalImportCount

    @property
    def successfulImportCount(self):
        return self.__st_successfulImportCount

    def addPostByYear(self, year):
        if year in self.__st_posts_per_year:
            self.__st_posts_per_year[year] += 1
        else:
            self.__st_posts_per_year[year] = 1

    def addPostByFullDate(self, date):
        if date in self.__st_posts_per_date:
            self.__st_posts_per_date[date] += 1
        else:
            self.__st_posts_per_date[date] = 1

    def increasePostCount(self):
        self.__st_totalPostsCount += 1

    def increaseCompaniesCount(self):
        self.__st_totalCompaniesCount += 1

    def increaseTotalImportCount(self):
        self.__st_totalImportCount += 1

    def increaseSuccessfulImportCount(self):
        self.__st_successfulImportCount += 1
