from datetime import datetime

logFilePath = "logs/"
logFileName = "messages_{}.log".format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
databasePath = "databases/stocker/stockerbot-export.xlsx"
workSheetName = "stockerbot-export"
databaseFileName = "stockerbot-export.xlsx"
stocksBasePath = "databases/stocks"
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
FINAL_DATABASE_TEXT_COLUMN = "Prediction"
FINAL_DATABASE_PREDICTION_COLUMN = "Tweet"
FINAL_DATABASE_FOLDER = "databases/final_data/"
FINAL_DATABASE_TRAIN = "train_tweets.csv"
FINAL_DATABASE_TEST = "test_tweets.csv"

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
