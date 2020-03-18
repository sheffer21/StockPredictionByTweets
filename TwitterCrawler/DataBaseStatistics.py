from common import constants as const
from TwitterCrawler.DataBaseOperationsService import DataBaseOperationsService as operation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class DataBaseStatistics:

    def __init__(self, logger):
        self.logger = logger

    def PublishDataBaseCompaniesGraph(self):
        self.logger.printAndLog(const.MessageType.Regular.value, "Plotting crawler companies statistics...")
        companiesValues = self.GetCompaniesDataCount()
        labels = list(key.split(" ", 2)[0] for key in companiesValues.keys())
        xCoordinates = np.arange(len(labels))
        heights = companiesValues.values()

        fig, ax = plt.subplots()
        ax.set_ylabel('Tweets')
        ax.set_title('Number of Tweets by Company')
        ax.set_xticks(xCoordinates)
        ax.set_xticklabels(labels)
        ax.bar(x=xCoordinates, height=heights, width=0.35, color=['red', 'green'])

        # Arrange labels
        for item in (ax.get_xticklabels()):
            item.set_fontsize(9)

        self.SavePlotToFile(const.twitterCrawlerCompaniesStatistics)

    def PublishDataBaseCompaniesKeywordsGraph(self):
        self.logger.printAndLog(const.MessageType.Regular.value, "Plotting crawler companies keywords statistics...")
        companiesKeywords, companiesPossibleKeywords = self.GetCompaniesKeywordsDataCount()
        labels = list(key.split(" ", 2)[0] for key in companiesKeywords.keys())
        xCoordinates = np.arange(len(labels))
        keywordsHeights = companiesKeywords.values()
        possibleKeywordsHeights = companiesPossibleKeywords.values()

        fig, ax = plt.subplots()
        width = 0.35
        rect1 = ax.bar(xCoordinates - width / 2, keywordsHeights, width, label=const.COMPANY_KEYWORDS_COLUMN)
        rect2 = ax.bar(xCoordinates + width / 2, possibleKeywordsHeights, width,
                       label=const.COMPANY_POSSIBLE_KEYWORDS_COLUMN)

        ax.set_ylabel('Tweets')
        ax.set_title('Number of Tweets by Company')
        ax.set_xticks(xCoordinates)
        ax.set_xticklabels(labels)
        ax.legend()

        DataBaseStatistics.autoLabel(rect1, ax)
        DataBaseStatistics.autoLabel(rect2, ax)

        # Arrange labels
        for item in (ax.get_xticklabels()):
            item.set_fontsize(9)

        self.SavePlotToFile(const.twitterCrawlerPossibleKeywordsStatistics)

    def SavePlotToFile(self, plotPath):
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(18, 10)
        if not os.path.exists(const.twitterCrawlerStatisticsFolder):
            os.makedirs(const.twitterCrawlerStatisticsFolder)

        plt.savefig(f"{const.twitterCrawlerStatisticsFolder}/{plotPath}", dpi=500)
        self.logger.printAndLog(const.MessageType.Regular.value, f"Saved plot {plotPath}")

    @staticmethod
    def autoLabel(rect, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rect:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    @staticmethod
    def GetCompaniesDataCount():
        dataBase = pd.read_csv(operation.GetMergedDataBaseFilePath())
        companiesValues = {}
        companies, _, _ = DataBaseStatistics.GetCompaniesKeywords()
        DataBaseStatistics.ResetCountDictionariesLabels(companiesValues, companies)
        for idx, row in dataBase.iterrows():
            companiesValues[row[const.COMPANY_COLUMN]] += 1

        return companiesValues

    @staticmethod
    def GetCompaniesKeywordsDataCount():
        dataBase = pd.read_csv(operation.GetMergedDataBaseFilePath())
        companies, keywords, possibleKeywords = DataBaseStatistics.GetCompaniesKeywords()
        companiesKeywords = {}
        DataBaseStatistics.ResetCountDictionariesLabels(companiesKeywords, companies)
        companiesPossibleKeywords = {}
        DataBaseStatistics.ResetCountDictionariesLabels(companiesPossibleKeywords, companies)
        for idx, row in dataBase.iterrows():
            keyword = row[const.SEARCH_KEYWORD_COLUMN]
            companyName = row[const.COMPANY_COLUMN]

            if (companyName in keywords and keyword in keywords[companyName]) or keyword == companyName:
                companiesKeywords[companyName] += 1
            if companyName in possibleKeywords and (keyword in possibleKeywords[companyName]):
                companiesPossibleKeywords[companyName] += 1

        return companiesKeywords, companiesPossibleKeywords

    @staticmethod
    def ResetCountDictionariesLabels(dictionary, labels):
        for label in labels:
            dictionary[label] = 0

    @staticmethod
    def GetCompaniesKeywords():
        companiesData = pd.read_csv(operation.GetCompaniesFilePath())
        keywords = {}
        possibleKeywords = {}
        companies = []
        for idx, company in companiesData.iterrows():
            companies.append(company[const.COMPANY_COLUMN])
            key = []
            if not pd.isnull(company[const.COMPANY_KEYWORDS_COLUMN]):
                key.extend(company[const.COMPANY_KEYWORDS_COLUMN].split(", "))
            if not pd.isnull(company[const.STOCK_SYMBOL_COLUMN]):
                key.extend(company[const.STOCK_SYMBOL_COLUMN].split(", "))

            keywords[company[const.COMPANY_COLUMN]] = key

            if not pd.isnull(company[const.COMPANY_POSSIBLE_KEYWORDS_COLUMN]):
                possibleKeywords[company[const.COMPANY_COLUMN]] = \
                    company[const.COMPANY_POSSIBLE_KEYWORDS_COLUMN].split(", ")

        return companies, keywords, possibleKeywords
