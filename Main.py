import os
import xlrd
import yfinance as yf
from Company import Company
from Post import Post
from StockInfo import StockInfo


# Project methods:
def openAndPrepareDatabase(path, sheetName):
    wb = xlrd.open_workbook(path)
    database = wb.sheet_by_name(sheetName)

    return database


def prepareLocalDatabase(database):
    for databaseRowIndex in range(1, database.nrows):
        postId = databaseRowIndex
        postText = database.cell_value(databaseRowIndex, POST_TEXT_COLUMN)
        postTimestamp = database.cell_value(databaseRowIndex, POST_TIMESTAMP_COLUMN)
        postSource = database.cell_value(databaseRowIndex, POST_SOURCE_COLUMN)
        postSymbols = database.cell_value(databaseRowIndex, POST_SYMBOLS_COLUMN)
        postCompany = database.cell_value(databaseRowIndex, POST_COMPANY_COLUMN)
        postUrl = database.cell_value(databaseRowIndex, POST_URL_COLUMN)
        postVerified = database.cell_value(databaseRowIndex, POST_VERIFIED_COLUMN)

        postCompaniesList = []
        postSymbolsParsed = postSymbols.split('-')
        postCompaniesParsed = postCompany.split('*')

        for companiesArrayIndex in range(len(postSymbolsParsed)):
            newCompany = Company(postSymbolsParsed[companiesArrayIndex], postCompaniesParsed[companiesArrayIndex])
            postCompaniesList.append(newCompany)
            companiesDict[postSymbolsParsed[companiesArrayIndex]] = postCompaniesParsed[companiesArrayIndex]

        newPost = Post(postId, postText, postTimestamp, postSource, postCompaniesList,
                       postUrl, postVerified, "unknown", "unknown")

        postsList.append(newPost)


def printLocalDatabase(maxIterations):
    iterationsCount = 0  # For debugging
    for post in postsList:
        print(post.description)
        iterationsCount += 1

        # For debugging
        if iterationsCount == maxIterations:
            break


def printCompaniesDict(maxIterations):
    iterationsCount = 0  # For debugging
    for companySymbol in companiesDict:
        print("Company symbol: {}, company name: {}".format(companySymbol, companiesDict[companySymbol]))
        iterationsCount += 1

        # For debugging
        if iterationsCount == maxIterations:
            break


def printDelimiter():
    print("-----------------------------------------------------------")


def getStockDataBySymbolAndDates(companyName, stockSymbol, infoStartDate, infoEndDate):
    filePath = "{}/{}_{}_{}.{}".format(stocksBasePath, stockSymbol, infoStartDate, infoEndDate, "csv")

    if not os.path.isfile(filePath):
        printStr = "Fetching data for company: {},\n " \
                   "\t with stock symbol: {},\n" \
                   "\t from date: {},\n" \
                   "\t to date: {}, " \
                   "".format(companyName, stockSymbol, infoStartDate, infoEndDate, filePath)

        print(printStr)

        data = yf.download(stockSymbol, infoStartDate, infoEndDate)  # returned data is 'DataFrame'
        if "True" == "True":  # TODO: check if return data is ok
            print("Fetching succeeded, saving to file: {}".format(filePath))
            export_csv = data.to_csv(filePath, index=None, header=True)
        else:
            print("Fetching failed...")

        printDelimiter()

    return filePath


def getStartDate(originalTimeStamp):
    # TODO: return timestamp
    return '2015-01-01'


def getEndDate(originalTimeStamp):
    # TODO: return timestamp
    return '2015-01-01'


def importStocksDatabasesForPosts():
    for post in postsList:
        # Fetch database for specified post
        postCompanies = post.companiesList
        postTimeStamp = post.timeStamp
        postDateStart = getStartDate(postTimeStamp)
        postDateEnd = getEndDate(postTimeStamp)

        for company in postCompanies:
            # Fetch database for each company the post effected on
            stockFilePath = getStockDataBySymbolAndDates(company.name,
                                                         company.stockSymbol,
                                                         postDateStart,
                                                         postDateEnd)

            newStockInfo = StockInfo(company.stockSymbol,
                                     postDateStart,
                                     postDateEnd,
                                     stockFilePath)

            post.addPostStockDatabase(newStockInfo)


# Project properties:
postsList = []
companiesDict = {}
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


# Project Main
def main():
    print("Welcome to our project :)")

    # Get Database
    print("Loading databases files...")
    database = openAndPrepareDatabase(databasePath, workSheetName)

    # Build local database
    print("Building local databases...")
    prepareLocalDatabase(database)

    # Importing stocks databases
    print("Importing stocks databases...")
    importStocksDatabasesForPosts()

    # TODO: Analyze stocks databases
    # TODO: Initial database process
    # TODO: NLP database process

    # Print database & companies (for debugging):
    # printLocalDatabase(printPostsLimit)
    # printCompaniesDict(printCompaniesLimit)


# Run project
main()
