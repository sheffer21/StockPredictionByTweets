from datetime import datetime
import os
import sys
import xlrd
import yfinance as yf
from pandas import DataFrame
import matplotlib.pyplot as plt
from Company import Company
from Post import Post
from StockInfo import StockInfo
import pandas_datareader.data as web
import pandas as pd
from iexfinance.stocks import Stock
import matplotlib.pyplot as plt
from iexfinance.stocks import get_historical_data
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from Statistics import Statistics


class Utility:
    postsList = []
    failedImports = {}
    companiesDict = {}
    logFilePath = "messages.log"
    logFileName = ""
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
    maxImportsAtOnce = float('inf')
    printPostsLimit = 10  # For debugging
    printCompaniesLimit = 10  # For debugging
    successfulImportsCount = 0
    totalImportsCount = 0
    importDaysBeforePostDate = 0
    importDaysAfterPostDate = 0
    statistics = Statistics(datetime.now())

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self, postsList, companiesDict, logFilePath, logFileName, databasePath, workSheetName, databaseFileName,
                 stocksBasePath, POST_ID_COLUMN, POST_TEXT_COLUMN, POST_TIMESTAMP_COLUMN, POST_SOURCE_COLUMN,
                 POST_SYMBOLS_COLUMN, POST_COMPANY_COLUMN, POST_URL_COLUMN, POST_VERIFIED_COLUMN, printPostsLimit,
                 printCompaniesLimit, maxImportsAtOnce, importDaysBeforePostDate, importDaysAfterPostDate, statistics):

        Utility.postsList = postsList
        Utility.companiesDict = companiesDict
        Utility.logFilePath = logFilePath
        Utility.logFileName = logFileName
        Utility.databasePath = databasePath
        Utility.workSheetName = workSheetName
        Utility.databaseFileName = databaseFileName
        Utility.stocksBasePath = stocksBasePath
        Utility.POST_ID_COLUMN = POST_ID_COLUMN
        Utility.POST_TEXT_COLUMN = POST_TEXT_COLUMN
        Utility.POST_TIMESTAMP_COLUMN = POST_TIMESTAMP_COLUMN
        Utility.POST_SOURCE_COLUMN = POST_SOURCE_COLUMN
        Utility.POST_SYMBOLS_COLUMN = POST_SYMBOLS_COLUMN
        Utility.POST_COMPANY_COLUMN = POST_COMPANY_COLUMN
        Utility.POST_URL_COLUMN = POST_URL_COLUMN
        Utility.POST_VERIFIED_COLUMN = POST_VERIFIED_COLUMN
        Utility.printPostsLimit = printPostsLimit  # For debugging
        Utility.printCompaniesLimit = printCompaniesLimit  # For debugging
        Utility.maxImportsAtOnce = maxImportsAtOnce
        Utility.successfulImportsCount = 0
        Utility.importDaysBeforePostDate = importDaysBeforePostDate
        Utility.importDaysAfterPostDate = importDaysAfterPostDate
        Utility.statistics = statistics

    @staticmethod
    def printAndLog(messageType, message):
        prefix = {
            "Regular": "",
            "Error": Utility.FAIL,
            "Success": Utility.OKGREEN,
            "Summarize": Utility.OKBLUE,
            "Header": Utility.HEADER
        }

        suffix = {
            "Regular": "",
            "Error": Utility.ENDC,
            "Success": Utility.ENDC,
            "Summarize": Utility.ENDC,
            "Header": Utility.ENDC
        }

        # Print to log:
        old_stdout = sys.stdout
        logFile = open(Utility.logFilePath + Utility.logFileName, "a")
        sys.stdout = logFile
        print(message)
        logFile.close()
        sys.stdout = old_stdout

        # Print to console:
        print(prefix.get(messageType, "") + message + suffix.get(messageType, ""))

    @staticmethod
    def openAndPrepareDatabase(path, sheetName):
        wb = xlrd.open_workbook(path)
        database = wb.sheet_by_name(sheetName)
        return database

    def prepareLocalDatabase(self, database):
        for databaseRowIndex in range(1, database.nrows):
            postId = databaseRowIndex
            postText = database.cell_value(databaseRowIndex, self.POST_TEXT_COLUMN)
            postTimestamp = database.cell_value(databaseRowIndex, self.POST_TIMESTAMP_COLUMN)
            postSource = database.cell_value(databaseRowIndex, self.POST_SOURCE_COLUMN)
            postSymbols = database.cell_value(databaseRowIndex, self.POST_SYMBOLS_COLUMN)
            postCompany = database.cell_value(databaseRowIndex, self.POST_COMPANY_COLUMN)
            postUrl = database.cell_value(databaseRowIndex, self.POST_URL_COLUMN)
            postVerified = database.cell_value(databaseRowIndex, self.POST_VERIFIED_COLUMN)

            postCompaniesList = []
            postSymbolsParsed = postSymbols.split('-')
            postCompaniesParsed = postCompany.split('*')

            for companiesArrayIndex in range(len(postSymbolsParsed)):
                newCompany = Company(postSymbolsParsed[companiesArrayIndex], postCompaniesParsed[companiesArrayIndex])
                postCompaniesList.append(newCompany)
                self.companiesDict[postSymbolsParsed[companiesArrayIndex]] = postCompaniesParsed[companiesArrayIndex]

            newPost = Post(postId, postText, postTimestamp, postSource, postCompaniesList,
                           postUrl, postVerified, "unknown", "unknown")

            self.postsList.append(newPost)

    def printLocalDatabase(self, maxIterations):
        iterationsCount = 0  # For debugging
        for post in self.postsList:
            self.printAndLog("Regular", post.description)
            iterationsCount += 1

            # For debugging
            if iterationsCount == maxIterations:
                break

    def printCompaniesDict(self, maxIterations):
        iterationsCount = 0  # For debugging
        for companySymbol in self.companiesDict:
            self.printAndLog("Regular", "Company symbol: {}, company name: {}\n"
                                        "".format(companySymbol, self.companiesDict[companySymbol]))

            iterationsCount += 1

            # For debugging
            if iterationsCount == maxIterations:
                break

    def printDelimiter(self):
        self.printAndLog("Regular", "-----------------------------------------------------------")

    # Exporting 'DataFrame' to csv
    @staticmethod
    def exportDataFrame(dataFrame, filePath):
        dataFrame.to_csv(filePath, index=None, header=True)

    @staticmethod
    def getStartDate(originalPostDate, originalPostTime):
        newDateTime = datetime(originalPostDate.year,
                               originalPostDate.month,
                               originalPostDate.day - Utility.importDaysBeforePostDate)
        return newDateTime

    @staticmethod
    def getEndDate(originalPostDate, originalPostTime):
        newDateTime = datetime(originalPostDate.year,
                               originalPostDate.month,
                               originalPostDate.day + Utility.importDaysAfterPostDate)
        return newDateTime

    @staticmethod
    def getStockDataBySymbolAndDates(stockSymbol, infoStartDate, infoEndDate):
        data = {}
        # data = web.DataReader(stockSymbol, 'yahoo', infoStartDate, infoEndDate)
        data = yf.download(stockSymbol, infoStartDate, infoEndDate, interval="1h")
        return data

    def printFailedImports(self):
        for failure in self.failedImports:
            self.printAndLog("Regular", "Failed to fetch: company symbol: {}, company name: {}"
                                        "".format(failure, self.failedImports[failure]))

    def getPostStocksFilePath(self, companyName, stockSymbol, infoStartDate, infoEndDate):
        filePath = "{}/{}_{}_{}.{}".format(self.stocksBasePath, stockSymbol, '2019-1-1', '2019-1-3', "csv")

        if not os.path.isfile(filePath):
            Utility.totalImportsCount += 1

            self.printAndLog("Regular", "Import tries count: {}".format(Utility.totalImportsCount))

            printStr = "Fetching data for company: {},\n " \
                       "\t with stock symbol: {},\n" \
                       "\t from date: {},\n" \
                       "\t to date: {}, " \
                       "".format(companyName, stockSymbol, infoStartDate, infoEndDate, filePath)

            self.printAndLog("Regular", printStr)

            try:
                dataFrame = self.getStockDataBySymbolAndDates(stockSymbol, infoStartDate, infoEndDate)
                if len(dataFrame) == 0:
                    self.printAndLog("Error", "Fetching failed...")
                    Utility.failedImports[stockSymbol] = companyName
                    return ""
            except:
                self.printAndLog("Error", "An error occurred while trying to fetch for stock symbol: {}".format(stockSymbol))
                return ""

            self.printAndLog("Success", "Fetching succeeded, saving to file: {}".format(filePath))
            self.exportDataFrame(dataFrame, filePath)
            Utility.successfulImportsCount += 1

            self.printDelimiter()

        return filePath

    def importStocksDatabasesForPosts(self):
        for post in self.postsList:
            # Fetch database for specified post
            postCompanies = post.companiesList
            postDateStart = self.getStartDate(post.date, post.time)
            postDateEnd = self.getEndDate(post.date, post.time)

            for company in postCompanies:
                # Fetch database for each company the post effected on
                stockFilePath = self.getPostStocksFilePath(company.name,
                                                           company.stockSymbol,
                                                           postDateStart,
                                                           postDateEnd)

                newStockInfo = StockInfo(company.stockSymbol,
                                         postDateStart,
                                         postDateEnd,
                                         stockFilePath)

                post.addPostStockDatabase(newStockInfo)

                if Utility.totalImportsCount == Utility.maxImportsAtOnce:
                    break

            if Utility.totalImportsCount == Utility.maxImportsAtOnce:
                break

        Utility.printAndLog("Summarize",
                            "Import done. {} passed out of {}.".format(Utility.successfulImportsCount,
                                                                       Utility.totalImportsCount))
