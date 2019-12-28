import os
from datetime import datetime, timedelta
from Common import Constants as const
import yfinance as yf
from Initializer.Company import Company
from Initializer.Post import Post
from Initializer.StockInfo import StockInfo
from Initializer.Statistics import Statistics
import pandas as pd
from sklearn.model_selection import train_test_split


class ProgramManager:
    postsList = []
    initialDatabase = {}
    companiesDict = {}
    failedSymbolsImports = {}
    statistics = Statistics(datetime.now())
    dirname = os.path.dirname(__file__)

    finalDatabase = pd.DataFrame(columns=[const.PREDICTION_COLUMN, const.TEXT_COLUMN])

    # filename = os.path.join(dirname, c.companies_path)

    def __init__(self, logger):
        if not os.path.isdir(os.path.join(ProgramManager.dirname, const.stocksBasePath)):
            os.mkdir(os.path.join(ProgramManager.dirname, const.stocksBasePath))

        self.logger = logger

    @staticmethod
    def openAndPrepareRawDatabase():
        databasePath = os.path.join(ProgramManager.dirname, const.databasePath)
        ProgramManager.initialDatabase = pd.read_csv(databasePath)

    @staticmethod
    def exportDataFrame(dataFrame, exportPath):
        dataFrame.to_csv(exportPath)

    @staticmethod
    def getStartDate(originalPostDate):
        newDateTime = originalPostDate - timedelta(days=const.importDaysBeforePostDate)
        return newDateTime

    @staticmethod
    def getEndDate(originalPostDate):
        newDateTime = originalPostDate + timedelta(days=const.importDaysAfterPostDate)
        return newDateTime

    @staticmethod
    def getStockDataBySymbolAndDates(stockSymbol, infoStartDate, infoEndDate):
        symbolDownloader = yf.Ticker(stockSymbol)
        data = symbolDownloader.history(start=infoStartDate, end=infoEndDate, actions="true")
        return data

    @property
    def database(self):
        return ProgramManager.finalDatabase

    def printLocalDatabase(self):
        iterationsCount = 0  # For debugging
        for post in self.postsList:
            self.logger.printAndLog(const.MessageType.Regular, post.description)
            iterationsCount += 1

            # For debugging
            if iterationsCount == const.printPostsLimit:
                break

    def printCompaniesDict(self):
        iterationsCount = 0  # For debugging
        for companySymbol in self.companiesDict:
            self.logger.printAndLog(const.MessageType.Regular, "Company symbol: {}, company name: {}\n"
                                                                     "".format(companySymbol, self.companiesDict[companySymbol]))

            iterationsCount += 1

            if iterationsCount == const.printCompaniesLimit:
                break

    def printDelimiter(self):
        self.logger.printAndLog(const.MessageType.Regular, "-----------------------------------------------------------")

    def printFailedImports(self):
        for failure in self.failedSymbolsImports:
            self.logger.printAndLog(const.MessageType.Regular, "Failed to fetch: company symbol: {}, company name: {}"
                                                                     "".format(failure, self.failedSymbolsImports[failure]))

    def getPostStocksFilePath(self, companyName, stockSymbol, infoStartDate, infoEndDate):
        filePath = "{}/{}_{}_{}.{}".format(const.stocksBasePath, stockSymbol, infoStartDate, infoEndDate, "csv")
        filePath = os.path.join(ProgramManager.dirname, filePath)

        if not os.path.isfile(filePath):
            ProgramManager.statistics.increaseTotalImportCount()
            self.logger.printAndLog(const.MessageType.Regular,
                                    "Import tries count: {}".format(ProgramManager.statistics.totalImportCount))

            printStr = "Fetching data for company: {},\n " \
                       "\t with stock symbol: {},\n" \
                       "\t from date: {},\n" \
                       "\t to date: {}, " \
                       "".format(companyName, stockSymbol, infoStartDate, infoEndDate, filePath)

            self.logger.printAndLog(const.MessageType.Regular, printStr)

            try:
                dataFrame = self.getStockDataBySymbolAndDates(stockSymbol, infoStartDate, infoEndDate)
                if len(dataFrame) == 0:
                    self.logger.printAndLog(const.MessageType.Error, "Fetching failed...")
                    ProgramManager.failedSymbolsImports[stockSymbol] = companyName
                    return ""
            except:
                self.logger.printAndLog(const.MessageType.Error,
                                        "An error occurred while trying to fetch for stock symbol: {}".format(stockSymbol))
                return ""

            self.logger.printAndLog(const.MessageType.Success, "Fetching succeeded, saving to file: {}".format(filePath))
            self.exportDataFrame(dataFrame, filePath)
            ProgramManager.statistics.increaseSuccessfulImportCount()

            self.printDelimiter()

        return filePath

    def prepareLocalDatabase(self):
        count = 0
        for databaseRow in ProgramManager.initialDatabase.values:
            count += 1
            postId = databaseRow[const.POST_ID_COLUMN]
            postText = databaseRow[const.POST_TEXT_COLUMN]
            postTimestamp = databaseRow[const.POST_TIMESTAMP_COLUMN]
            postSource = databaseRow[const.POST_SOURCE_COLUMN]
            postSymbols = databaseRow[const.POST_SYMBOLS_COLUMN]
            postCompany = databaseRow[const.POST_COMPANY_COLUMN]
            postUrl = databaseRow[const.POST_URL_COLUMN]
            postVerified = databaseRow[const.POST_VERIFIED_COLUMN]

            postCompaniesList = []
            postSymbolsParsed = postSymbols.split('-')
            postCompaniesParsed = postCompany.split('*')

            for companiesArrayIndex in range(len(postSymbolsParsed)):
                newCompany = Company(postSymbolsParsed[companiesArrayIndex], postCompaniesParsed[companiesArrayIndex])
                postCompaniesList.append(newCompany)
                self.companiesDict[postSymbolsParsed[companiesArrayIndex]] = postCompaniesParsed[companiesArrayIndex]

            newPost = Post(postId, postText, postTimestamp, postSource, postCompaniesList, postUrl, postVerified)

            self.postsList.append(newPost)

    def importStocksDatabasesForPosts(self):
        for post in self.postsList:
            # Fetch database for specified post
            postCompanies = post.companiesList
            postDateStart = self.getStartDate(post.date)
            postDateEnd = self.getEndDate(post.date)

            for company in postCompanies:
                # Fetch database for each company the post effected on
                stockFilePath = self.getPostStocksFilePath(company.name,
                                                           company.stockSymbol,
                                                           postDateStart,
                                                           postDateEnd)

                if not stockFilePath == "":
                    newStockInfo = StockInfo(company.name,
                                             company.stockSymbol,
                                             post.date,
                                             postDateStart,
                                             postDateEnd,
                                             stockFilePath)

                    post.addStockInfo(company.stockSymbol, newStockInfo)
                    self.logger.printAndLog(const.MessageType.Regular,
                                            "Saved stock info for symbol: {}.".format(company.stockSymbol))
                else:
                    self.logger.printAndLog(const.MessageType.Error,
                                            "Could not save stock info for symbol: {}.".format(company.stockSymbol))

                if ProgramManager.statistics.totalImportCount == const.maxImportsAtOnce:
                    break

            if ProgramManager.statistics.totalImportCount == const.maxImportsAtOnce:
                break

        self.logger.printAndLog(const.MessageType.Summarize, "Import done. {} passed out of {}.".format(
            ProgramManager.statistics.successfulImportCount,
            ProgramManager.statistics.totalImportCount))

    def add_false_stocks_to_data_base(self):
        count = 0
        result = 0
        for post in self.postsList:
            stock_info = StockInfo("", "", "", "", "", "")
            stock_info.finalResult = result
            post.addStockInfo("Symbol", stock_info)
            count += 1
            if count % 50 == 0:
                result += 0.1

    @staticmethod
    def save_split_data_base_to_csv():
        train, test = train_test_split(ProgramManager.finalDatabase, test_size=0.2, random_state=42)

        # reset indices
        train.reset_index(drop=True)
        test.reset_index(drop=True)
        train.to_csv(f'{const.FINAL_DATABASE_FOLDER}{const.TrainFile}', index=False)
        test.to_csv(f'{const.FINAL_DATABASE_FOLDER}{const.testFile}', index=False)

    def build_final_database(self):
        ProgramManager.finalDatabase = pd.concat(
            [pd.DataFrame([[stockInfo.finalResult, post.text]],
                          columns=[const.PREDICTION_COLUMN,
                                   const.TEXT_COLUMN])
             for post in self.postsList
             for stockInfo in post.stocksInfo.values()], ignore_index=True)

        ProgramManager.save_split_data_base_to_csv()
