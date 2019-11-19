import os
import sys
from datetime import datetime
import Utility
from Statistics import Statistics

# Project properties:
postsList = []
companiesDict = {}
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
maxImportsAtOnce = float('inf')
importDaysBeforePostDate = 2
importDaysAfterPostDate = 3
statistics = Statistics(datetime.now())


# Project Main
def main():
    util = Utility.Utility(postsList, companiesDict, logFilePath, logFileName, databasePath, workSheetName, databaseFileName,
                           stocksBasePath, POST_ID_COLUMN, POST_TEXT_COLUMN, POST_TIMESTAMP_COLUMN, POST_SOURCE_COLUMN,
                           POST_SYMBOLS_COLUMN, POST_COMPANY_COLUMN, POST_URL_COLUMN, POST_VERIFIED_COLUMN, printPostsLimit,
                           printCompaniesLimit, maxImportsAtOnce, importDaysBeforePostDate, importDaysAfterPostDate, statistics)

    util.printAndLog("Summarize", "Welcome to our project :)")
    util.printAndLog("Regular", "Log path: {}{}\n".format(logFilePath, logFileName))

    # Get Database
    util.printAndLog("Header", "Loading databases files...")
    database = util.openAndPrepareDatabase(databasePath, workSheetName)

    # Build local database
    util.printAndLog("Header", "Building local databases...")
    util.prepareLocalDatabase(database)

    # TODO: add more databases?

    # Importing stocks databases
    util.printAndLog("Header", "Importing stocks databases...")
    util.importStocksDatabasesForPosts()

    # Debug:
    util.printFailedImports()

    # TODO: Analyze stocks databases
    # TODO: NLP database process

    # Print database & companies (for debugging):
    # printLocalDatabase(printPostsLimit)
    # printCompaniesDict(printCompaniesLimit)
    # print(statistics.getStatistics())


# Run project
main()
