import tweepy
import pandas as pd
from datetime import datetime
import constants as c
from logger import Logger as Log
import os
from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results

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

maximum_search_size = 100


def crawl_twitter():
    Log.print_and_log(c.MessageType.Header, "Start crawling Twitter...")

    # Connect to twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
        Log.print_and_log(c.MessageType.Success, "Twitter authentication OK")
    except:
        Log.print_and_log(c.MessageType.Error, "Error during twitter authentication")

    # Get new tweets
    companies_data = pd.read_csv(c.companies_path)
    result = pd.DataFrame()
    for company in companies_data.values:
        result = pd.concat([result, search_twitter(company[0], company[1], api)])

    Log.print_and_log(c.MessageType.Regular, "Finish twitter search")

    # Save new tweets to file
    save_tweets_to_file(result)


def search_twitter(keyword, company_symbol, api):
    date = (datetime.now()).strftime("%Y-%m-%d")
    search_word = f"#{keyword.lower()}"
    Log.print_and_log(c.MessageType.Regular, f"Searching tweets for {keyword}")
    results = api.search(q=search_word, lang="en", result_type='popular', until=date, rpp=maximum_search_size)
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
                          columns=columns)
             for tweet in results])


def save_tweets_to_file(tweets):
    date_with_hour = datetime.now().strftime("%d-%m-%Y-%H%M")
    tweets_file = f'{c.twitterCrawlerDataBaseDir}{c.twitterCrawlerFilesPrefix}{date_with_hour}.csv'

    if not os.path.isdir(c.twitterCrawlerDataBaseDir):
        os.mkdir(c.twitterCrawlerDataBaseDir)

    tweets.to_csv(tweets_file, index=False)
    Log.print_and_log(c.MessageType.Success, f"Saved new tweets to file {tweets_file}")

