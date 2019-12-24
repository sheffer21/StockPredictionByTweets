import os
import json
import xlrd
from datetime import datetime, timedelta
from common import constants as c
import yfinance as yf
from Company import Company
from Post import Post
from StockInfo import StockInfo
from Statistics import Statistics
import pandas as pd
from sklearn.model_selection import train_test_split
from common.logger import Logger as Log


class ProgramManager:
    with open('config.json') as config_file:
        configuration = json.load(config_file)

    postsList = []
    initial_database = {}
    companiesDict = {}
    failedSymbolsImports = {}
    statistics = Statistics(datetime.now())

    finalDatabase = pd.DataFrame(columns=[c.PREDICTION_COLUMN, c.TEXT_COLUMN])

    def __init__(self):
        if not os.path.isdir(c.stocksBasePath):
            os.mkdir(c.stocksBasePath)

    @staticmethod
    def openAndPrepareRawDatabase():
        wb = xlrd.open_workbook(c.databasePath, on_demand=True)
        ProgramManager.initial_database = wb.sheet_by_name(c.workSheetName)

    @staticmethod
    def exportDataFrame(dataFrame, exportPath):
        dataFrame.to_csv(exportPath)

    @staticmethod
    def getStartDate(originalPostDate):
        newDateTime = originalPostDate - timedelta(days=c.importDaysBeforePostDate)
        return newDateTime

    @staticmethod
    def getEndDate(originalPostDate):
        newDateTime = originalPostDate + timedelta(days=c.importDaysAfterPostDate)
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
            Log.print_and_log(c.MessageType.Regular.value, post.description)
            iterationsCount += 1

            # For debugging
            if iterationsCount == c.printPostsLimit:
                break

    def printCompaniesDict(self):
        iterationsCount = 0  # For debugging
        for companySymbol in self.companiesDict:
            Log.print_and_log(c.MessageType.Regular.value, "Company symbol: {}, company name: {}\n"
                                        "".format(companySymbol, self.companiesDict[companySymbol]))

            iterationsCount += 1

            if iterationsCount == c.printCompaniesLimit:
                break

    def printDelimiter(self):
        Log.print_and_log(c.MessageType.Regular.value, "-----------------------------------------------------------")

    def printFailedImports(self):
        for failure in self.failedSymbolsImports:
            Log.print_and_log(c.MessageType.Regular.value, "Failed to fetch: company symbol: {}, company name: {}"
                                        "".format(failure, self.failedSymbolsImports[failure]))

    def getPostStocksFilePath(self, companyName, stockSymbol, infoStartDate, infoEndDate):
        filePath = "{}/{}_{}_{}.{}".format(c.stocksBasePath, stockSymbol, infoStartDate, infoEndDate, "csv")

        if not os.path.isfile(filePath):
            ProgramManager.statistics.increaseTotalImportCount()
            Log.print_and_log(c.MessageType.Regular.value,
                              "Import tries count: {}".format(ProgramManager.statistics.totalImportCount))

            printStr = "Fetching data for company: {},\n " \
                       "\t with stock symbol: {},\n" \
                       "\t from date: {},\n" \
                       "\t to date: {}, " \
                       "".format(companyName, stockSymbol, infoStartDate, infoEndDate, filePath)

            Log.print_and_log(c.MessageType.Regular.value, printStr)

            try:
                dataFrame = self.getStockDataBySymbolAndDates(stockSymbol, infoStartDate, infoEndDate)
                if len(dataFrame) == 0:
                    Log.print_and_log(c.MessageType.Error.value, "Fetching failed...")
                    ProgramManager.failedSymbolsImports[stockSymbol] = companyName
                    return ""
            except:
                Log.print_and_log(c.MessageType.Error.value,
                                  "An error occurred while trying to fetch for stock symbol: {}".format(stockSymbol))
                return ""

            Log.print_and_log(c.MessageType.Success.value, "Fetching succeeded, saving to file: {}".format(filePath))
            self.exportDataFrame(dataFrame, filePath)
            ProgramManager.statistics.increaseSuccessfulImportCount()

            self.printDelimiter()

        return filePath

    def prepareLocalDatabase(self):
        for databaseRowIndex in range(1, ProgramManager.initial_database.nrows):
            postId = databaseRowIndex
            postText = ProgramManager.initial_database.cell_value(databaseRowIndex, c.POST_TEXT_COLUMN)
            postTimestamp = ProgramManager.initial_database.cell_value(databaseRowIndex, c.POST_TIMESTAMP_COLUMN)
            postSource = ProgramManager.initial_database.cell_value(databaseRowIndex, c.POST_SOURCE_COLUMN)
            postSymbols = ProgramManager.initial_database.cell_value(databaseRowIndex, c.POST_SYMBOLS_COLUMN)
            postCompany = ProgramManager.initial_database.cell_value(databaseRowIndex, c.POST_COMPANY_COLUMN)
            postUrl = ProgramManager.initial_database.cell_value(databaseRowIndex, c.POST_URL_COLUMN)
            postVerified = ProgramManager.initial_database.cell_value(databaseRowIndex, c.POST_VERIFIED_COLUMN)

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
                    Log.print_and_log(c.MessageType.Regular.value,
                                      "Saved stock info for symbol: {}.".format(company.stockSymbol))
                else:
                    Log.print_and_log(c.MessageType.Error.value,
                                      "Could not save stock info for symbol: {}.".format(company.stockSymbol))

                if ProgramManager.statistics.totalImportCount == c.maxImportsAtOnce:
                    break

            if ProgramManager.statistics.totalImportCount == c.maxImportsAtOnce:
                break

        Log.print_and_log(c.MessageType.Summarize.value, "Import done. {} passed out of {}.".format(
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
        train.to_csv(f'{c.FINAL_DATABASE_FOLDER}{c.TrainFile}', index=False)
        test.to_csv(f'{c.FINAL_DATABASE_FOLDER}{c.testFile}', index=False)

    def build_final_database(self):
        ProgramManager.finalDatabase = pd.concat(
            [pd.DataFrame([[stockInfo.finalResult, post.text]],
                          columns=[c.PREDICTION_COLUMN,
                                   c.TEXT_COLUMN])
             for post in self.postsList
             for stockInfo in post.stocksInfo.values()], ignore_index=True)

        ProgramManager.save_split_data_base_to_csv()

