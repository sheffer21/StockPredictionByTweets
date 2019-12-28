import tweepy
import pandas as pd
from datetime import datetime
from Common import Constants as c
import os
from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results


class TwitterCrawler:

    # TODO : move to secrets file
    consumer_key = '991t554wZOqeYfIAWOxg4pRID'
    consumer_secret = 'PbQpvVZGfETYkMK4FWyv9pNPCb10lZwrrpPg3BRnoPm4u9JMh1'
    access_token = '1196704751641137152-IX9GPWlE1sJBX0q9PTk9yP8yAsiYIq'
    access_token_secret = 'HTE8WwpYWGbhKzvDihB28FEAMbdPfM075rF6eEwla5G7j'

    columns = [c.DATE_COLUMN,
               c.ID_COLUMN,
               c.TWEET_COLUMN,
               c.USER_ID_COLUMN,
               c.USER_NAME_COLUMN,
               c.USER_SCREEN_NAME_COLUMN,
               c.USER_LOCATION_COLUMN,
               c.USER_URL_COLUMN,
               c.USER_DESCRIPTION_COLUMN,
               c.PLACE_COLUMN,
               c.ENTITIES_COLUMN,
               c.STOCK_SYMBOL_COLUMN]

    maximum_search_size = c.maximumSearchSize

    @staticmethod
    def crawlTwitter(logger):
        logger.printAndLog(c.MessageType.Header, "Start crawling Twitter...")

        # Connect to twitter
        auth = tweepy.OAuthHandler(TwitterCrawler.consumer_key, TwitterCrawler.consumer_secret)
        auth.set_access_token(TwitterCrawler.access_token, TwitterCrawler.access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        try:
            api.verify_credentials()
            logger.printAndLog(c.MessageType.Success, "Twitter authentication OK")
        except:
            logger.printAndLog(c.MessageType.Error, "Error during twitter authentication")

        # Get new Tweets
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, c.companies_path)
        companies_data = pd.read_csv(filename)
        result = pd.DataFrame()
        for company in companies_data.values:
            result = pd.concat([result, TwitterCrawler.searchTwitter(logger, company[0], company[1], api)])

        logger.printAndLog(c.MessageType.Regular, "Finish twitter search")

        # Save new Tweets to file
        TwitterCrawler.saveTweetsToFile(logger, result)

    @staticmethod
    def searchTwitter(logger, keyword, company_symbol, api):
        date = (datetime.now()).strftime("%Y-%m-%d")
        searchWord = f"#{keyword.lower()}"
        logger.printAndLog(c.MessageType.Regular, f"Searching tweets for {keyword}")
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

    @staticmethod
    def saveTweetsToFile(logger, tweets):
        date_with_hour = datetime.now().strftime("%d-%m-%Y-%H%M")
        dirname = os.path.dirname(__file__)
        tweetsFile = f'{c.twitterCrawlerDataBaseDir}{c.twitterCrawlerFilesPrefix}{date_with_hour}.csv'
        tweetsFile = os.path.join(dirname, tweetsFile)

        if not os.path.isdir(c.twitterCrawlerDataBaseDir):
            os.mkdir(os.path.join(dirname, c.twitterCrawlerDataBaseDir))

        tweets.to_csv(tweetsFile, index=False)
        logger.printAndLog(c.MessageType.Success, f"Saved new tweets to file {tweetsFile}")
