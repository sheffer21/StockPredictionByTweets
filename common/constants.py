import enum

# Program paths
databasePath = "databases/stocker/stockerbot-export.xlsx"
workSheetName = "stockerbot-export"
databaseFileName = "stockerbot-export.xlsx"
stocksBasePath = "databases/stocks"
companies_path = 'companies.csv'
FINAL_DATABASE_FOLDER = "databases/final_data/"
TrainFile = "train_tweets.csv"
testFile = "test_tweets.csv"
twitterCrawlerDataBaseDir = "databases/"
twitterCrawlerFilesPrefix = "tweets-"
twitterCrawlerMergedFilesName = "tweets-Merged.csv"

POST_ID_COLUMN = 0
POST_TEXT_COLUMN = 1
POST_TIMESTAMP_COLUMN = 2
POST_SOURCE_COLUMN = 3
POST_SYMBOLS_COLUMN = 4
POST_COMPANY_COLUMN = 5
POST_URL_COLUMN = 6
POST_VERIFIED_COLUMN = 7
printPostsLimit = 10  # For debugging
printCompaniesLimit = 10  # For debugging
maxImportsAtOnce = 10
importDaysBeforePostDate = 30
importDaysAfterPostDate = 30

# Data base column names
PREDICTION_COLUMN = "Prediction"
TEXT_COLUMN = "Tweet"
DATE_COLUMN = "Date"
ID_COLUMN = "ID"
TWEET_COLUMN = "Tweet"
USER_ID_COLUMN = "User id"
USER_NAME_COLUMN = "User name"
USER_SCREEN_NAME_COLUMN = "User screen name"
USER_LOCATION_COLUMN = "User location"
USER_URL_COLUMN = "User url"
USER_DESCRIPTION_COLUMN = "User description"
PLACE_COLUMN = "place"
ENTITIES_COLUMN = "Entities"
STOCK_SYMBOL_COLUMN = "Stock Symbol"
COMPANY_COLUMN = "Company"


class MessageType(enum.Enum):
    Regular = 1
    Header = 2
    Summarize = 3
    printLog = 4
    Error = 5
    Success = 6
