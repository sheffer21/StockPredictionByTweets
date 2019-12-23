import tweepy
import pandas as pd
from datetime import datetime

consumer_key = '991t554wZOqeYfIAWOxg4pRID'
consumer_secret = 'PbQpvVZGfETYkMK4FWyv9pNPCb10lZwrrpPg3BRnoPm4u9JMh1'
access_token = '1196704751641137152-IX9GPWlE1sJBX0q9PTk9yP8yAsiYIq'
access_token_secret = 'HTE8WwpYWGbhKzvDihB28FEAMbdPfM075rF6eEwla5G7j'


def crawl_twitter():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
        print("Twitter authentication OK")
    except:
        print("Error during twitter authentication")

    columns = ["Date",
               "ID",
               "Text",
               "User id",
               "User name",
               "User screen name",
               "User location",
               "User url",
               "User description",
               "place",
               "Entities",
               "Stock Symbol"]

    companies_data = pd.read_csv('databases/stocker/companies.csv')
    result = pd.DataFrame()
    date = datetime.now().strftime("%Y-%m-%d")
    for company in companies_data.values():
        result = pd.concat([result, pd.concat(
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
                            company[1]]],
                          columns=columns)
             for tweet in api.search(q=company[0], lang="en", result_type='popular', until=date, rpp=100)])])

    date = datetime.now().strftime("%d-%m-%Y-%H%M")
    result.to_csv(f'databases/stocker/tweets-{date}.csv', index=False)


