import csv
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from Common import Constants as const


def ListToFormattedString(alist):
    format_list = ['\'{:>3}\'' for item in alist]
    s = ', '.join(format_list)
    return s.format(*alist)


class StockInfo:
    columnColor = {
        "Open": "red",
        "High": "black",
        "Low": "yellow",
        "Close": "blue",
        "Volume": "green",
        "Dividends": "pink",
        "Stock Splits": "purple"
    }

    columnIdx = {
        "Open": 0,
        "High": 1,
        "Low": 2,
        "Close": 3,
        "Volume": 4,
        "Dividends": 5,
        "Stock Splits": 6
    }

    dirname = os.path.dirname(__file__)
    graphDaysInterval = const.graphDaysInterval
    effectiveDaysAfterPost = const.effectiveDaysAfterPost
    effectiveColumnIndex = columnIdx.get(const.effectiveColumnName)

    def __init__(self, stockCompany, stockSymbol, postOriginalDate, infoStartDate, infoEndDate, rawDataPath):
        self.__stockCompany = stockCompany
        self.__s_stockSymbol = stockSymbol
        self.__s_postOriginalDate = postOriginalDate
        self.__s_infoStartDate = infoStartDate
        self.__s_infoEndDate = infoEndDate
        self.__s_dataPath = rawDataPath
        self.__s_stockData = {}
        self.__s_deltaData = {}
        self.__s_percentChanges = {}
        self.__s_deviationPerDate = {}
        self.__s_averageChanges = []
        self.__s_averageValues = []
        self.__s_stockTag = 0
        self.__s_graph_plot = None

        self.parseData()
        self.fillMissingData()
        self.rearrangeData()
        self.analyzeStock()
        self.calculateDeviations()
        self.calculateTag()

        self.createPlotByColumnNames("Values", ["Open", "Close"], "yes")
        self.createPlotByColumnNames("Change", ["Close"], "yes")

    @property
    def stockTag(self):
        return self.__s_stockTag

    def parseData(self):
        with open(self.__s_dataPath, newline='') as dataFile:
            reader = csv.DictReader(dataFile)
            for row in reader:
                rowKey = row['Date']
                rowValue = [
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Volume']),
                    float(row['Dividends']),
                    float(row['Stock Splits'])
                ]

                self.__s_stockData[rowKey] = rowValue

    def getPreviousAvailableDate(self, date):

        dateFormat = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        prevDate = str(dateFormat - datetime.timedelta(days=1))
        while prevDate not in self.__s_stockData:
            prevDateFormat = datetime.datetime.strptime(prevDate, '%Y-%m-%d').date()
            prevDate = str(prevDateFormat - datetime.timedelta(days=1))

        return prevDate

    def fillBetween(self, missingDates, fromDate, toDate):

        fromDateFormat = datetime.datetime.strptime(fromDate, '%Y-%m-%d').date()
        toDateFormat = datetime.datetime.strptime(toDate, '%Y-%m-%d').date()
        toFillDate = fromDateFormat + datetime.timedelta(days=1)
        if toFillDate == toDateFormat:
            return

        if str(fromDateFormat) in self.__s_stockData.keys():
            fromVec = self.__s_stockData[str(fromDateFormat)]
        else:
            fromVec = missingDates[str(fromDateFormat)]

        toVec = self.__s_stockData[str(toDateFormat)]
        fillVec = [x / 2 for x in np.add(fromVec, toVec)]
        missingDates[str(toFillDate)] = fillVec

        self.fillBetween(missingDates, str(toFillDate), str(toDateFormat))

    def fillMissingData(self):

        missingDates = {}
        isFirst = True
        for currDate in self.__s_stockData:
            postDate = datetime.datetime.strptime(currDate, '%Y-%m-%d').date()
            prev_1_Date = str(postDate - datetime.timedelta(days=1))

            if isFirst or prev_1_Date in self.__s_stockData.keys():  # Nothing is missing
                isFirst = False
                continue

            lastDate = self.getPreviousAvailableDate(currDate)
            self.fillBetween(missingDates, lastDate, currDate)

        for date in missingDates:
            self.__s_stockData[date] = missingDates[date]

    def rearrangeData(self):
        arrangedDatabase = {}
        date = next(iter(self.__s_stockData))
        while date in self.__s_stockData.keys():
            arrangedDatabase[date] = self.__s_stockData[date]
            dateFormat = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            nextDateFormat = dateFormat + datetime.timedelta(days=1)
            date = str(nextDateFormat)

        self.__s_stockData = arrangedDatabase

    def analyzeStock(self):
        rowsCount = 0
        rowsSum = np.zeros(7, )

        for row in self.__s_stockData:
            postDate = datetime.datetime.strptime(row, '%Y-%m-%d').date()
            prevDate = str(postDate - datetime.timedelta(days=1))
            postDate = str(postDate)

            if prevDate not in self.__s_stockData:
                continue

            rowsSum = np.add(rowsSum, np.array(self.__s_stockData[row]))
            rowsCount += 1

            percentVec = []
            prevVec = np.array(self.__s_stockData[prevDate])
            currVec = np.array(self.__s_stockData[postDate])
            diffVec = currVec - prevVec

            for i in range(len(currVec)):
                if diffVec[i] == 0 or prevVec[i] == 0:
                    percentVec.append(0)
                else:
                    percentVec.append(diffVec[i] * 100 / prevVec[i])

            self.__s_deltaData[postDate] = diffVec
            self.__s_percentChanges[postDate] = percentVec

        averageCount = 0
        averageRowsSum = np.zeros(7, )
        for row in self.__s_percentChanges:
            averageRowsSum = np.add(averageRowsSum, np.array(self.__s_percentChanges[row]))
            averageCount += 1

        self.__s_averageChanges = averageRowsSum / averageCount
        self.__s_averageValues = rowsSum / rowsCount

    def calculateDeviations(self):
        deviation = {}
        for date in self.__s_percentChanges:
            averageExcluded = self.calculateAverageExcluded(date)
            deviation[date] = self.__s_percentChanges[date][StockInfo.effectiveColumnIndex] - averageExcluded

        self.__s_deviationPerDate = deviation

    def calculateAverageExcluded(self, indexDate):
        sum = 0
        count = 0
        for date in self.__s_percentChanges:
            if date == indexDate:
                continue
            count += 1
            sum += self.__s_percentChanges[date][StockInfo.effectiveColumnIndex]

        if count == 0:
            return 0

        return sum / count

    def calculateTag(self):
        tag = 0
        index = 0
        for currentDate in self.__s_deviationPerDate:

            currentDateInDateFormat = datetime.datetime.strptime(currentDate, '%Y-%m-%d').date()
            if currentDateInDateFormat < self.__s_postOriginalDate:
                continue

            tag += (StockInfo.effectiveDaysAfterPost - index) * self.__s_deviationPerDate[currentDate]
            if index == (StockInfo.effectiveDaysAfterPost - 1):
                break

            index += 1

        sumOfDays = 0
        for count in range(StockInfo.effectiveDaysAfterPost):
            sumOfDays += count

        self.__s_stockTag = tag

    def createPlotByColumnNames(self, plotType, columnNames, showPlot="no"):
        x = []
        y = {}
        for columnName in columnNames:
            y[columnName] = []

        x_ticks = []

        if plotType == "Values":
            plotData = self.__s_stockData
            titlePrefix = ""
            yTitle = "Value"
        else:
            plotData = self.__s_percentChanges
            titlePrefix = "change of"
            yTitle = "Change of value in %"

        count = 0
        originalDateIndex = 0
        for rowDate in plotData:
            postDate = datetime.datetime.strptime(rowDate, '%Y-%m-%d')
            if count % StockInfo.graphDaysInterval == 0 or postDate.date() == self.__s_postOriginalDate:
                x_ticks.append(rowDate)
                if postDate.date() == self.__s_postOriginalDate:
                    originalDateIndex = len(x_ticks) - 1
                    count = 0

            count += 1

            x.append(rowDate)
            for columnName in columnNames:
                y[columnName].append(plotData[rowDate][StockInfo.columnIdx.get(columnName)])

        fig, ax = plt.subplots()
        plt.grid()

        for y_key in y:
            ax.plot(x, y[y_key], color=StockInfo.columnColor.get(y_key), label='\'{}\' change value'.format(y_key))
            ax.get_xticklabels()[originalDateIndex].set_color("red")

            if len(columnNames) == 1:
                averageVector = np.empty(len(x))

                if plotType == "Values":
                    averageValue = self.__s_averageValues[StockInfo.columnIdx.get(y_key)]
                else:
                    averageValue = self.__s_averageChanges[StockInfo.columnIdx.get(y_key)]

                averageVector.fill(averageValue)
                ax.plot(x, averageVector, 'green')
                ax.text(0.01, 0.03, 'Average value: {0:.4f}'.format(averageValue), transform=ax.transAxes)

        plt.xlabel('Date')
        plt.ylabel(yTitle)
        plt.xticks(x_ticks, rotation=45, ha='right')
        plt.title('{} {} values for company \'{}\'\n Between dates: {} - {}\n Post publish date: {}'
                  ''.format(ListToFormattedString(columnNames),
                            titlePrefix,
                            self.__stockCompany,
                            self.__s_infoStartDate,
                            self.__s_infoEndDate,
                            self.__s_postOriginalDate))

        plt.legend(loc="upper left")
        plt.tight_layout()

        graphsDirectory = os.path.join(StockInfo.dirname, const.graphsPath)
        if not os.path.isdir(graphsDirectory):
            os.mkdir(graphsDirectory)

        filePath = "{}/{}_{}_{}_{}_{}.{}".format(graphsDirectory, const.graphFileName, plotType, self.__s_stockSymbol,
                                                 self.__s_infoStartDate, self.__s_infoEndDate, "png")
        plt.savefig(filePath)

        if showPlot == "yes":
            plt.show()

    def plotAllSeparately(self):
        self.createPlotByColumnNames("Values", "Open")
        self.createPlotByColumnNames("Values", "High")
        self.createPlotByColumnNames("Values", "Low")
        self.createPlotByColumnNames("Values", "Close")
        self.createPlotByColumnNames("Values", "Volume")
