from datetime import datetime
import os
import sys
import xlrd
import yfinance as yf
from Company import Company
from Post import Post
from StockInfo import StockInfo
from Statistics import Statistics
import json


class ProgramManager:
    with open('config.json') as config_file:
        configuration = json.load(config_file)

    postsList = []
    database = {}
    companiesDict = {}
    failedImports = {}
    logFilePath = configuration['logFilePath']
    logFileName = configuration['logFileName'].format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    databasePath = configuration['databasePath']
    workSheetName = configuration['workSheetName']
    databaseFileName = configuration['databaseFileName']
    stocksBasePath = configuration['stocksBasePath']
    POST_ID_COLUMN = configuration['POST_ID_COLUMN']
    POST_TEXT_COLUMN = configuration['POST_TEXT_COLUMN']
    POST_TIMESTAMP_COLUMN = configuration['POST_TIMESTAMP_COLUMN']
    POST_SOURCE_COLUMN = configuration['POST_SOURCE_COLUMN']
    POST_SYMBOLS_COLUMN = configuration['POST_SYMBOLS_COLUMN']
    POST_COMPANY_COLUMN = configuration['POST_COMPANY_COLUMN']
    POST_URL_COLUMN = configuration['POST_URL_COLUMN']
    POST_VERIFIED_COLUMN = configuration['POST_VERIFIED_COLUMN']
    printPostsLimit = configuration['printPostsLimit']  # For debugging
    printCompaniesLimit = configuration['printCompaniesLimit']  # For debugging
    maxImportsAtOnce = configuration['maxImportsAtOnce']
    importDaysBeforePostDate = configuration['importDaysBeforePostDate']
    importDaysAfterPostDate = configuration['importDaysAfterPostDate']
    statistics = Statistics(datetime.now())

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def __init__(self):
        if not os.path.isdir(ProgramManager.logFilePath):
            os.mkdir(ProgramManager.logFilePath)

        if not os.path.isdir(ProgramManager.stocksBasePath):
            os.mkdir(ProgramManager.stocksBasePath)

    @staticmethod
    def printMessage(messageType, message):
        prefix = {
            "Regular": "",
            "Error": ProgramManager.FAIL,
            "Success": ProgramManager.OKGREEN,
            "Summarize": ProgramManager.OKBLUE,
            "Header": ProgramManager.HEADER
        }

        suffix = {
            "Regular": "",
            "Error": ProgramManager.ENDC,
            "Success": ProgramManager.ENDC,
            "Summarize": ProgramManager.ENDC,
            "Header": ProgramManager.ENDC
        }

        if messageType == "printLog":
            print(prefix.get("Regular", "") +
                  "Log path: {}{}\n".format(ProgramManager.logFilePath, ProgramManager.logFileName) +
                  suffix.get(messageType, ""))
        else:
            print(prefix.get(messageType, "") + message + suffix.get(messageType, ""))

    @staticmethod
    def printAndLog(messageType, message):
        # Print to console:
        ProgramManager.printMessage(messageType, message)

        # Print to log:
        old_stdout = sys.stdout
        logFile = open(ProgramManager.logFilePath + ProgramManager.logFileName, "a")
        sys.stdout = logFile
        ProgramManager.printMessage(messageType, message)
        logFile.close()
        sys.stdout = old_stdout

    @staticmethod
    def openAndPrepareDatabase():
        wb = xlrd.open_workbook(ProgramManager.databasePath)
        ProgramManager.database = wb.sheet_by_name(ProgramManager.workSheetName)

    def prepareLocalDatabase(self):
        for databaseRowIndex in range(1, ProgramManager.database.nrows):
            postId = databaseRowIndex
            postText = ProgramManager.database.cell_value(databaseRowIndex, self.POST_TEXT_COLUMN)
            postTimestamp = ProgramManager.database.cell_value(databaseRowIndex, self.POST_TIMESTAMP_COLUMN)
            postSource = ProgramManager.database.cell_value(databaseRowIndex, self.POST_SOURCE_COLUMN)
            postSymbols = ProgramManager.database.cell_value(databaseRowIndex, self.POST_SYMBOLS_COLUMN)
            postCompany = ProgramManager.database.cell_value(databaseRowIndex, self.POST_COMPANY_COLUMN)
            postUrl = ProgramManager.database.cell_value(databaseRowIndex, self.POST_URL_COLUMN)
            postVerified = ProgramManager.database.cell_value(databaseRowIndex, self.POST_VERIFIED_COLUMN)

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
                               originalPostDate.day - ProgramManager.importDaysBeforePostDate)
        return newDateTime

    @staticmethod
    def getEndDate(originalPostDate, originalPostTime):
        newDateTime = datetime(originalPostDate.year,
                               originalPostDate.month,
                               originalPostDate.day + ProgramManager.importDaysAfterPostDate)
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
            ProgramManager.statistics.increaseTotalImportCount()
            self.printAndLog("Regular", "Import tries count: {}".format(ProgramManager.statistics.totalImportsCount))

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
                    ProgramManager.failedImports[stockSymbol] = companyName
                    return ""
            except:
                self.printAndLog("Error", "An error occurred while trying to fetch for stock symbol: {}".format(stockSymbol))
                return ""

            self.printAndLog("Success", "Fetching succeeded, saving to file: {}".format(filePath))
            self.exportDataFrame(dataFrame, filePath)
            ProgramManager.statistics.increaseSuccessfulImportCount()

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

                post.addStockInfo(company.stockSymbol, newStockInfo)

                if ProgramManager.statistics.totalImportCount == ProgramManager.maxImportsAtOnce:
                    break

            if ProgramManager.statistics.totalImportCount == ProgramManager.maxImportsAtOnce:
                break

        ProgramManager.printAndLog("Summarize",
                                   "Import done. {} passed out of {}.".format(ProgramManager.statistics.successfulImportCount,
                                                                              ProgramManager.statistics.totalImportCount))
