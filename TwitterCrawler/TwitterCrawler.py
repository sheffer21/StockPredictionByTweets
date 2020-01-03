import tweepy
import pandas as pd
from datetime import datetime
import constants as const
import os
from DataBaseOperationsService import DataBaseOperationsService as operations


class TwitterCrawler:
    consumer_key = '991t554wZOqeYfIAWOxg4pRID'
    consumer_secret = 'PbQpvVZGfETYkMK4FWyv9pNPCb10lZwrrpPg3BRnoPm4u9JMh1'
    access_token = '1196704751641137152-IX9GPWlE1sJBX0q9PTk9yP8yAsiYIq'
    access_token_secret = 'HTE8WwpYWGbhKzvDihB28FEAMbdPfM075rF6eEwla5G7j'

    columns = [const.DATE_COLUMN,
               const.ID_COLUMN,
               const.TEXT_COLUMN,
               const.USER_ID_COLUMN,
               const.USER_NAME_COLUMN,
               const.USER_SCREEN_NAME_COLUMN,
               const.USER_LOCATION_COLUMN,
               const.USER_URL_COLUMN,
               const.USER_DESCRIPTION_COLUMN,
               const.PLACE_COLUMN,
               const.ENTITIES_COLUMN,
               const.STOCK_SYMBOL_COLUMN,
               const.COMPANY_COLUMN,
               const.SEARCH_KEYWORD_COLUMN]

    maximumSearchSize = 100
    maximumNumberOfRequestsPerWindow = 180

    def __init__(self, logger):
        self.logger = logger
        self.numberOfSearches = 0
        self.api = ""

    # noinspection PyBroadException
    def crawlTwitter(self):
        self.logger.printAndLog(const.MessageType.Header, "Start crawling Twitter...")

        # Connect to twitter
        self.connectToTwitter()

        # Get new Tweets
        companiesFilePath = operations.GetCompaniesFilePath()
        companies_data = pd.read_csv(companiesFilePath)

        results = []
        for index, company in companies_data.iterrows():
            for keyword in TwitterCrawler.GetCompanyKeywords(company):
                res = self.searchTwitterForKeyword(keyword, company[0], company[1])
                if res is not None:
                    results.append(res)

        if not results:
            return

        data = pd.concat([result for result in results])
        data = operations.DropDataDuplicates(data, const.ID_COLUMN)

        # Get users followers
        self.addUsersFollowers(data)

        self.logger.printAndLog(const.MessageType.Regular, f"Total number of searches: {self.numberOfSearches}")
        self.logger.printAndLog(const.MessageType.Regular, "Finish twitter search")

        # Save new Tweets to file
        self.saveTweetsToFile(data)

    def addUsersFollowers(self, tweets):

        self.logger.printAndLog(const.MessageType.Regular, "Looking for users followers...")

        if self.api is "":
            self.connectToTwitter()

        # get unique user ids
        usersID = tweets[const.USER_ID_COLUMN].unique()

        # Create users id's to followers dictionary
        idToFollowers = {}
        for _id in usersID:
            try:
                self.logger.printAndLog(const.MessageType.Regular.value, f'Looking for {_id} followers count')
                user = self.api.get_user(user_id=_id)
            except:
                self.logger.printAndLog(const.MessageType.Error.value, f"User {_id} doesnt exists")

            idToFollowers[_id] = user.followers_count

        # Add followers numbers to tweets
        operations.AddColumnToDataByReferenceColumnValue(tweets,
                                                         idToFollowers,
                                                         const.USER_ID_COLUMN,
                                                         const.USER_FOLLOWERS_COLUMN)

    def connectToTwitter(self):
        auth = tweepy.OAuthHandler(TwitterCrawler.consumer_key, TwitterCrawler.consumer_secret)
        auth.set_access_token(TwitterCrawler.access_token, TwitterCrawler.access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        try:
            api.verify_credentials()
            self.logger.printAndLog(const.MessageType.Success, "Twitter authentication OK")
        except:
            self.logger.printAndLog(const.MessageType.Error, "Error during twitter authentication")

        self.api = api

    def searchTwitterForKeyword(self, keyword, company_name, company_symbol):
        searchWord = f"{keyword.lower().strip()}"
        self.logger.printAndLog(
            const.MessageType.Regular.value, f'Searching tweets containing "{keyword}" for Company {company_name}')

        date = datetime.now().strftime("%Y-%m-%d")
        resultsTypes = ['popular', 'mixed', 'recent']
        results = []
        for resultType in resultsTypes:
            result = self.searchInTwitter(searchWord, resultType, date)
            if result and len(result) != 0:
                results.append(result)

        if len(results) == 0:
            return None

        data = pd.concat(
            [pd.DataFrame([[tweet.created_at,
                            tweet.id_str,
                            tweet.text,
                            tweet.user.id,
                            tweet.user.name,
                            tweet.user.screen_name,
                            tweet.user.location,
                            tweet.user.url,
                            tweet.user.description,
                            tweet.place,
                            tweet.entities,
                            company_symbol,
                            company_name,
                            keyword]],
                          columns=TwitterCrawler.columns)
             for result in results
             for tweet in result])

        data = operations.DropDataDuplicates(data, const.ID_COLUMN)
        return data

    def searchInTwitter(self, searchWord, resultType, date):
        self.numberOfSearches += 1

        try:
            return self.api.search(q=searchWord, lang="en", result_type=resultType, until=date,
                                   rpp=TwitterCrawler.maximumSearchSize)
        except:
            self.logger.printAndLog(const.MessageType.Error.value, f'Fail to search for {searchWord} '
            f'with results type {resultType} till date {date}')

    def saveTweetsToFile(self, tweets):
        date_with_hour = datetime.now().strftime("%d-%m-%Y-%H%M")
        dirName = os.path.dirname(__file__)
        tweetsFile = f'{const.twitterCrawlerDataBaseDir}{const.twitterCrawlerFilesPrefix}{date_with_hour}.csv'
        tweetsFile = os.path.join(dirName, tweetsFile)

        if not os.path.isdir(const.twitterCrawlerDataBaseDir):
            os.mkdir(os.path.join(dirName, const.twitterCrawlerDataBaseDir))

        operations.SaveToCsv(tweets, tweetsFile)
        self.logger.printAndLog(const.MessageType.Success, f"Saved new tweets to file {tweetsFile}")

    @staticmethod
    def GetCompanyKeywords(company):
        keywords = [company[const.COMPANY_COLUMN]]

        TwitterCrawler.ExtendList(keywords, company[const.COMPANY_KEYWORDS_COLUMN])
        TwitterCrawler.ExtendList(keywords, company[const.COMPANY_POSSIBLE_KEYWORDS_COLUMN])

        return keywords

    @staticmethod
    def ExtendList(list_, values):
        if not pd.isnull(values):
            list_.extend(values.split(", "))
