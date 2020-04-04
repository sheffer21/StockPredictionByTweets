import os
from datetime import datetime, timedelta
from common import constants as const
import yfinance as yf
from Preprocessor.Company import Company
from Preprocessor.Post import Post
from Preprocessor.StockInfo import StockInfo
from Preprocessor.Statistics import Statistics
import pandas as pd
from sklearn.model_selection import train_test_split
import re


class PreProcessor:
    postsList = []
    initialDatabase = pd.DataFrame()
    companiesDict = {}
    failedSymbolsImports = pd.DataFrame(columns=['company', 'symbol', 'from', 'to'])
    finalDatabase = pd.DataFrame(columns=[const.PREDICTION_COLUMN, const.TEXT_COLUMN])
    statistics = Statistics(datetime.now())
    dirName = os.path.dirname(__file__)

    def __init__(self, logger):
        if not os.path.isdir(os.path.join(PreProcessor.dirName, const.stocksBasePath)):
            os.mkdir(os.path.join(PreProcessor.dirName, const.stocksBasePath))

        self.logger = logger

    @staticmethod
    def openAndPrepareRawDatabase():
        databasePath = os.path.join(
            PreProcessor.dirName, f"{const.databaseLocationPath}/{const.twitterCrawlerMergedFilesName}")
        PreProcessor.initialDatabase = pd.read_csv(databasePath)

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
        return PreProcessor.finalDatabase

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
                                                               "".format(companySymbol,
                                                                         self.companiesDict[companySymbol]))

            iterationsCount += 1

            if iterationsCount == const.printCompaniesLimit:
                break

    def printDelimiter(self):
        self.logger.printAndLog(const.MessageType.Regular,
                                "-----------------------------------------------------------")

    def printAndExportFailedImports(self):
        failedImportsDirectory = os.path.join(PreProcessor.dirName, const.failedImportsPath)
        if not os.path.isdir(failedImportsDirectory):
            os.mkdir(failedImportsDirectory)

        failedImports = PreProcessor.failedSymbolsImports
        for index, failure in failedImports.iterrows():
            self.logger.printAndLog(const.MessageType.Error, "Failed to fetch: company {}, symbol: {}, "
                                                             "from date: {}, to date: {}"
                                                             "".format(failure['company'], failure['symbol'],
                                                                       failure['from'], failure['to']))

        filePath = "{}/{}_{}.{}".format(failedImportsDirectory, const.failedImportsFileName,
                                        self.logger.getLoggerDate(), "csv")
        PreProcessor.failedSymbolsImports.to_csv(f'{filePath}', index=False)

    # noinspection PyBroadException
    def getPostStocksFilePath(self, companyName, stockSymbol, infoStartDate, infoEndDate):
        filePath = "{}/{}_{}_{}.{}".format(const.stocksBasePath, stockSymbol, infoStartDate, infoEndDate, "csv")
        filePath = os.path.join(PreProcessor.dirName, filePath)

        if not os.path.isfile(filePath):
            PreProcessor.statistics.increaseTotalImportCount()
            self.logger.printAndLog(const.MessageType.Regular,
                                    "Import tries count: {}".format(PreProcessor.statistics.totalImportCount))

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

                    failedImports = PreProcessor.failedSymbolsImports
                    newFailedImport = pd.DataFrame(data=[[companyName, stockSymbol, infoStartDate, infoEndDate]],
                                                   columns=['company', 'symbol', 'from', 'to'])
                    PreProcessor.failedSymbolsImports = failedImports.append(newFailedImport)
                    return ""
            except:
                self.logger.printAndLog(const.MessageType.Error,
                                        "An error occurred while trying to fetch for stock symbol: {}".format(
                                            stockSymbol))
                return ""

            self.logger.printAndLog(const.MessageType.Success,
                                    "Fetching succeeded, saving to file: {}".format(filePath))
            self.exportDataFrame(dataFrame, filePath)
            PreProcessor.statistics.increaseSuccessfulImportCount()

            self.printDelimiter()

        return filePath

    # noinspection PyTypeChecker
    def prepareLocalDatabase(self):
        for idx, databaseRow in PreProcessor.initialDatabase.iterrows():

            # Update after database changes
            postId = databaseRow[const.ID_COLUMN]
            postText = databaseRow[const.TEXT_COLUMN]
            postDate = databaseRow[const.DATE_COLUMN]
            postUserFollowers = databaseRow[const.USER_FOLLOWERS_COLUMN]
            postSymbols = databaseRow[const.STOCK_SYMBOL_COLUMN]
            postCompany = databaseRow[const.COMPANY_COLUMN]
            postKeyword = databaseRow[const.COMPANY_COLUMN]

            postCompaniesList = []
            postSymbolsParsed = postSymbols.split('-')
            postCompaniesParsed = postCompany.split('*')

            for companiesArrayIndex in range(len(postSymbolsParsed)):
                newCompany = Company(postSymbolsParsed[companiesArrayIndex], postCompaniesParsed[companiesArrayIndex])
                postCompaniesList.append(newCompany)
                self.companiesDict[postSymbolsParsed[companiesArrayIndex]] = postCompaniesParsed[companiesArrayIndex]

            newPost = Post(postId, postText, postDate, postUserFollowers, postCompaniesList, postKeyword)

            self.postsList.append(newPost)

    def importStocksDatabasesForPosts(self):
        for post in self.postsList:
            # Fetch database for specified post
            postCompanies = post.companiesList
            postDateStart = self.getStartDate(post.date)
            postDateEnd = self.getEndDate(post.date)

            for company in postCompanies:
                for symbol in self.GetCompanySymbols(company):
                    # Fetch database for each company the post effected on
                    stockFilePath = self.getPostStocksFilePath(company.name,
                                                               symbol,
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
                        # self.logger.printAndLog(const.MessageType.Regular,
                        #                        "Saved stock info for symbol: {}.".format(company.stockSymbol))
                    else:
                        self.logger.printAndLog(const.MessageType.Error,
                                                "Could not save stock info for symbol: {}.".format(company.stockSymbol))

                    # if PreProcessor.statistics.totalImportCount == const.maxImportsAtOnce:
                    #     break

            # if PreProcessor.statistics.totalImportCount == const.maxImportsAtOnce:
            #    break

        self.logger.printAndLog(const.MessageType.Summarize, "Import done. {} passed out of {}.".format(
            PreProcessor.statistics.successfulImportCount,
            PreProcessor.statistics.totalImportCount))

    @staticmethod
    def saveSplitDataBaseToCsv():
        train, test = train_test_split(PreProcessor.finalDatabase, test_size=0.2, random_state=42)

        databaseDirectory = os.path.join(PreProcessor.dirName, const.finalDatabaseFolder)
        if not os.path.isdir(databaseDirectory):
            os.mkdir(databaseDirectory)

        # reset indices
        train.reset_index(drop=True)
        test.reset_index(drop=True)
        train.to_csv(f'{databaseDirectory}{const.trainFile}', index=False)
        test.to_csv(f'{databaseDirectory}{const.testFile}', index=False)

    def buildFinalDatabase(self):
        PreProcessor.finalDatabase = pd.concat(
            [pd.DataFrame([[post.timeStamp,
                            stockInfo.stockTag,
                            self.clean_post(post.text),
                            ','.join(c.name for c in post.companiesList),
                            post.followers,
                            post.postKeyword]],
                          columns=[const.DATE_COLUMN,
                                   const.PREDICTION_COLUMN,
                                   const.TEXT_COLUMN,
                                   const.COMPANY_COLUMN,
                                   const.USER_FOLLOWERS_COLUMN,
                                   const.SEARCH_KEYWORD_COLUMN])
             for post in self.postsList
             for stockInfo in post.stocksInfo.values()
             if type(post.text) is str], ignore_index=True)

        PreProcessor.saveSplitDataBaseToCsv()

    @staticmethod
    def GetCompanySymbols(company):
        return company.stockSymbol.split(", ")

    @staticmethod
    def clean_post(text):
        # Get rid of urls
        text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', text)

        # Get rid of all non letters or numbers
        text = re.sub(r'[^A-Za-z0-9.]+', ' ', text)

        return text.strip()
