import tweepy
import pandas as pd
from datetime import datetime
from Common import Constants as const
import os
from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results


class TwitterCrawler:
    # TODO : move to secrets file
    consumer_key = '991t554wZOqeYfIAWOxg4pRID'
    consumer_secret = 'PbQpvVZGfETYkMK4FWyv9pNPCb10lZwrrpPg3BRnoPm4u9JMh1'
    access_token = '1196704751641137152-IX9GPWlE1sJBX0q9PTk9yP8yAsiYIq'
    access_token_secret = 'HTE8WwpYWGbhKzvDihB28FEAMbdPfM075rF6eEwla5G7j'

    columns = [const.DATE_COLUMN,
               const.ID_COLUMN,
               const.TWEET_COLUMN,
               const.USER_ID_COLUMN,
               const.USER_NAME_COLUMN,
               const.USER_SCREEN_NAME_COLUMN,
               const.USER_LOCATION_COLUMN,
               const.USER_URL_COLUMN,
               const.USER_DESCRIPTION_COLUMN,
               const.PLACE_COLUMN,
               const.ENTITIES_COLUMN,
               const.STOCK_SYMBOL_COLUMN]

    maximum_search_size = const.maximumSearchSize

    def __init__(self, logger):
        self.logger = logger

    # noinspection PyBroadException
    def crawlTwitter(self):
        self.logger.printAndLog(const.MessageType.Header, "Start crawling Twitter...")

        # Connect to twitter
        auth = tweepy.OAuthHandler(TwitterCrawler.consumer_key, TwitterCrawler.consumer_secret)
        auth.set_access_token(TwitterCrawler.access_token, TwitterCrawler.access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        try:
            api.verify_credentials()
            self.logger.printAndLog(const.MessageType.Success, "Twitter authentication OK")
        except:
            self.logger.printAndLog(const.MessageType.Error, "Error during twitter authentication")

        # Get new Tweets
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, const.companies_path)
        companies_data = pd.read_csv(filename)
        result = pd.DataFrame()
        for company in companies_data.values:
            result = pd.concat([result, self.searchTwitter(company[0], company[1], api)])

        self.logger.printAndLog(const.MessageType.Regular, "Finish twitter search")

        # Save new Tweets to file
        self.saveTweetsToFile(result)

    def searchTwitter(self, keyword, company_symbol, api):
        date = (datetime.now()).strftime("%Y-%m-%d")
        searchWord = f"#{keyword.lower()}"
        self.logger.printAndLog(const.MessageType.Regular, f"Searching tweets for {keyword}")
        results = api.search(q=searchWord, lang="en", result_type='popular', until=date, rpp=TwitterCrawler.maximum_search_size)
        if len(results) == 0:
            return

        return pd.concat(
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
                            company_symbol]],
                          columns=TwitterCrawler.columns)
             for tweet in results])

    def saveTweetsToFile(self, tweets):
        date_with_hour = datetime.now().strftime("%d-%m-%Y-%H%M")
        dirname = os.path.dirname(__file__)
        tweetsFile = f'{const.twitterCrawlerDataBaseDir}{const.twitterCrawlerFilesPrefix}{date_with_hour}.csv'
        tweetsFile = os.path.join(dirname, tweetsFile)

        if not os.path.isdir(const.twitterCrawlerDataBaseDir):
            os.mkdir(os.path.join(dirname, const.twitterCrawlerDataBaseDir))

        tweets.to_csv(tweetsFile, index=False)
        self.logger.printAndLog(const.MessageType.Success, f"Saved new tweets to file {tweetsFile}")

    def merge(self):
        self.logger.printAndLog(const.MessageType.Regular, "Merging database files...")
        dirname = os.path.dirname(__file__)
        mergePath = f'{const.twitterCrawlerDataBaseDir}{const.twitterCrawlerMergedFilesName}'
        mergePath = os.path.join(dirname, mergePath)

        old_merge_count = 0
        if os.path.exists(mergePath):
            old_merge = pd.read_csv(mergePath)
            old_merge_count = old_merge.shape[0]

        result = pd.DataFrame()

        dirPath = os.path.join(dirname, const.twitterCrawlerDataBaseDir)

        # Load Tweets from all files
        for file in os.listdir(dirPath):
            if file == const.twitterCrawlerMergedFilesName:
                break

            path = os.path.join(dirPath, file)
            self.logger.printAndLog(const.MessageType.Regular, f"Loading file: {path}")
            result = pd.concat([result, pd.read_csv(path)], sort=True)

        # Get all the unique Tweets
        unique_result = result.drop_duplicates()
        unique_result.to_csv(mergePath, index=False)
        self.logger.printAndLog(const.MessageType.Regular,
                                f"Added additional {unique_result.shape[0] - old_merge_count} to merge file")
        self.logger.printAndLog(const.MessageType.Success, "Successfully created a final merged database of Tweets")
