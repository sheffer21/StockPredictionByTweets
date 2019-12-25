import csv
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np


def ListToFormattedString(alist):
    format_list = ['\'{:>3}\'' for item in alist]
    s = ', '.join(format_list)
    return s.format(*alist)


class StockInfo:
    with open('config.json') as config_file:
        configuration = json.load(config_file)

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

    graphDaysInterval = configuration['graphDaysInterval']
    effectiveDaysAfterPost = configuration['effectiveDaysAfterPost']
    effectiveColumnIndex = columnIdx.get(configuration['effectiveColumnName'])

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
        self.__s_deviationOfDay = {}
        self.__s_averageChanges = []
        self.__s_averageValues = []
        self.__s_stockTag = 0

        self.parseData()
        self.fillMissingData()
        self.analyzeStock()
        self.calculateDeviation()
        self.calculateTag()

        print(self.__s_stockTag)

        self.plotByColumnNames("Values", ["Open", "Close"])
        self.plotByColumnNames("Change", ["Close"])

        if self.__s_stockSymbol == "FB":
            exit()

        @property
        def stockTag(self):
            return self.__s_stockTag

    @finalResult.setter
    def finalResult(self, result):
        self.__s_finalResult = result

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

    def plotAllSeparately(self):
        self.plotByColumnNames("Values", "Open")
        self.plotByColumnNames("Values", "High")
        self.plotByColumnNames("Values", "Low")
        self.plotByColumnNames("Values", "Close")
        self.plotByColumnNames("Values", "Volume")

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

        avrgCount = 0
        avrgRowsSum = np.zeros(7, )
        for row in self.__s_percentChanges:
            avrgRowsSum = np.add(avrgRowsSum, np.array(self.__s_percentChanges[row]))
            avrgCount += 1

        self.__s_averageChanges = avrgRowsSum / avrgCount
        self.__s_averageValues = rowsSum / rowsCount

    def plotByColumnNames(self, plotType, columnNames):
        x = []
        y = {}
        for columnName in columnNames:
            y[columnName] = []

        x_ticks = []

        if plotType == "Values":
            plotData = self.__s_stockData
        else:
            plotData = self.__s_percentChanges

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
        plt.ylabel('Change of value in %')
        plt.xticks(x_ticks, rotation=45)
        plt.title('{} change of values for company \'{}\'\n Between dates: {} - {}\n Post publish date: {}'
                  ''.format(ListToFormattedString(columnNames),
                            self.__stockCompany,
                            self.__s_infoStartDate,
                            self.__s_infoEndDate,
                            self.__s_postOriginalDate))

        plt.legend(loc="upper left")
        plt.show()
        # sys.exit()

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

    def calculateDeviation(self):

        deviation = {}
        for indexDate in self.__s_percentChanges:
            averageExcluded = self.calculateAverageExcluded(indexDate)
            deviation[indexDate] = self.__s_percentChanges[indexDate][StockInfo.effectiveColumnIndex] - averageExcluded

        print(self.__s_percentChanges)
        print(averageExcluded)
        print(deviation)

        self.__s_deviationOfDay = deviation

    def calculateTag(self):
        tag = 0
        index = 0
        for currentDate in self.__s_deviationOfDay:

            currentDateInDateFormat = datetime.datetime.strptime(currentDate, '%Y-%m-%d').date()
            if currentDateInDateFormat < self.__s_postOriginalDate:
                continue

            tag += (StockInfo.effectiveDaysAfterPost - index) * self.__s_deviationOfDay[currentDate]
            if index == (StockInfo.effectiveDaysAfterPost - 1):
                break

            index += 1

        sumOfDays = 0
        for count in range(StockInfo.effectiveDaysAfterPost):
            sumOfDays += count

        self.__s_stockTag = tag
