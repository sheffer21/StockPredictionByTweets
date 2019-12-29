import enum

# Global configurations
logsDirectory = "Log"
logNamePrefix = "messages_"
originalDatabasePath = "../Databases/stockerbot-export.csv"
databaseDateFormat = "%a %b %d %H:%M:%S +0000 %Y"
stocksBasePath = "../Databases/Stocks"
failedImportsPath = "../Databases/Stocks/FailedImports"
failedImportsFileName = "failedImports"
graphsPath = "../Databases/Graphs"
graphFileName = "plot"
companiesPath = '../Databases/Companies.csv'
finalDatabaseFolder = "../Databases/Database/"
trainFile = "train_tweets.csv"
testFile = "test_tweets.csv"

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

POST_ID_COLUMN = 0
POST_TEXT_COLUMN = 1
POST_TIMESTAMP_COLUMN = 2
POST_SOURCE_COLUMN = 3
POST_SYMBOLS_COLUMN = 4
POST_COMPANY_COLUMN = 5
POST_URL_COLUMN = 6
POST_VERIFIED_COLUMN = 7

# Twitter crawler configurations
twitterCrawlerDataBaseDir = "../Databases/Tweets/"
twitterCrawlerMergedFilesName = "Tweets-Merged.csv"
twitterCrawlerFilesPrefix = "Tweets-"
maximumSearchSize = 10

# Pre processing configurations
printPostsLimit = 10  # For debugging
printCompaniesLimit = 10  # For debugging
maxImportsAtOnce = 10
importDaysBeforePostDate = 30
importDaysAfterPostDate = 30
graphDaysInterval = 3
effectiveDaysAfterPost = 6
effectiveColumnName = "Open"

# Machine learning configurations

# Main program console colors
MAIN_HEADER = '\033[95m'
MAIN_ENDC = '\033[0m'


# Logger console colors
class MessageType(enum.Enum):
    Regular = 1
    Header = 2
    Summarize = 3
    printLog = 4
    Error = 5
    Success = 6
