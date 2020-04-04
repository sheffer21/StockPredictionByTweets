import enum

# Global configurations
logsDirectory = "Log"
logNamePrefix = "messages_"
databaseFolder = "../Databases/"
databaseLocationPath = "../Databases/Tweets/"
databaseDateFormat = "%Y-%m-%d %H:%M:%S"
stocksBasePath = "../Databases/Stocks"
failedImportsPath = "../Databases/Stocks/FailedImports"
failedImportsFileName = "failedImports"
graphsPath = "../Databases/Graphs"
graphFileName = "plot"
companiesPath = '../Databases/Companies.csv'
finalDatabaseFolder = "../Databases/Database/"
trainFile = "train_tweets.csv"
testFile = "test_tweets.csv"
trainFileDebug = "train_tweetsDebug.csv"
testFileDebug = "test_tweetsDebug.csv"

# Data base column names

# Tweets Data base
DATE_COLUMN = "Date"
ID_COLUMN = "ID"
TEXT_COLUMN = "Tweet"
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
SEARCH_KEYWORD_COLUMN = "Search Keyword"
USER_FOLLOWERS_COLUMN = "User followers"

# Final data base
PREDICTION_COLUMN = "Prediction"
# TEXT_COLUMN = "Tweet"

# Companies data file
# COMPANY_COLUMN = "Company"
COMPANY_KEYWORDS_COLUMN = "Keywords"
COMPANY_POSSIBLE_KEYWORDS_COLUMN = "Possible Keywords"

# Twitter crawler configurations
twitterCrawlerDataBaseDir = "../Databases/Tweets/"
twitterCrawlerMergedFilesName = "Tweets-Merged.csv"
twitterCrawlerTrailDataFilesName = "Tweets-TrailData.csv"
twitterCrawlerFilesPrefix = "Tweets-"
twitterCrawlerStatisticsFolder = "../TwitterCrawler/Statistics"
twitterCrawlerCompaniesStatistics = "TweetsByCompany.png"
twitterCrawlerPossibleKeywordsStatistics = "TweetsByPossibleKeywords.png"

# Pre processing configurations
printPostsLimit = 10  # For debugging
printCompaniesLimit = 20  # For debugging
maxImportsAtOnce = 10
importDaysBeforePostDate = 30
importDaysAfterPostDate = 30
graphDaysInterval = 3
effectiveDaysAfterPost = 6
effectiveColumnName = "Open"

# Machine learning configurations
MachineLearnerStatisticsFolder = "../MachineLearner/Statistics"
MachineLearnerTrainPlot = "Train_Plot"
MachineLearnerFollowersPlot = "Followers_Plot"
MachineLearnerLabelsPlot = "Labels_Plot"
TrainedModelDirectory = "../MachineLearner/TrainedModel/"
TrainedModelFile = "Trained_Model"

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
