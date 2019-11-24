import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys


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

    def __init__(self, stockCompany, stockSymbol, postOriginalDate, infoStartDate, infoEndDate, rawDataPath):
        self.__stockCompany = stockCompany
        self.__s_stockSymbol = stockSymbol
        self.__s_postOriginalDate = postOriginalDate
        self.__s_infoStartDate = infoStartDate
        self.__s_infoEndDate = infoEndDate
        self.__s_dataPath = rawDataPath
        self.__s_stockData = {}
        self.__s_finalResult = 0

        self.parseData()
        self.fillMissingData()
        self.analyzeStock()
        self.plotByColumnNames(["Open"])

    @property
    def finalResult(self):
        return self.__s_finalResult

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

    def fillMissingData(self):
        newStockData = {}

        for currDate in self.__s_stockData:
            postDate = datetime.datetime.strptime(currDate, '%Y-%m-%d').date()
            prev_1_Date = str(postDate - datetime.timedelta(days=1))
            prev_2_Date = str(postDate - datetime.timedelta(days=2))
            prev_3_Date = str(postDate - datetime.timedelta(days=3))
            postDate = str(postDate)

            if newStockData == {} or prev_1_Date in newStockData.keys():  # Nothing is missing
                newStockData[postDate] = self.__s_stockData[postDate]
            else:
                if prev_2_Date in newStockData.keys():  # Need to fill just prev day
                    firstVec = self.__s_stockData[prev_2_Date]
                    secondVec = self.__s_stockData[postDate]
                    mat = np.array([firstVec, secondVec])
                    filledVec = np.mean(mat, axis=0)
                    newStockData[prev_1_Date] = filledVec
                    newStockData[postDate] = self.__s_stockData[postDate]

                else:  # Need to fill 2 days
                    firstVec = self.__s_stockData[prev_3_Date]
                    secondVec = self.__s_stockData[postDate]
                    mat = np.array([firstVec, secondVec])
                    firstFilledVec = np.mean(mat, axis=0)
                    mat = np.array([firstFilledVec, secondVec])
                    secondFilledVec = np.mean(mat, axis=0)
                    newStockData[prev_2_Date] = firstFilledVec
                    newStockData[prev_1_Date] = secondFilledVec
                    newStockData[postDate] = self.__s_stockData[postDate]

        self.__s_stockData = newStockData

    def plotByColumnNames(self, columnNames):
        x = []
        y = {}
        for columnName in columnNames:
            y[columnName] = []

        x_ticks = []

        count = 0
        originalDateIndex = 0
        for rowDate in self.__s_stockData:
            postDate = datetime.datetime.strptime(rowDate, '%Y-%m-%d')
            if count % 3 == 0 or postDate.date() == self.__s_postOriginalDate:
                x_ticks.append(rowDate)
                if postDate.date() == self.__s_postOriginalDate:
                    originalDateIndex = len(x_ticks) - 1

            count += 1

            x.append(rowDate)
            for columnName in columnNames:
                y[columnName].append(self.__s_stockData[rowDate][StockInfo.columnIdx.get(columnName)])

        fig, ax = plt.subplots()
        plt.grid()

        for y_key in y:
            ax.plot(x, y[y_key], color=StockInfo.columnColor.get(y_key), label='\'{}\' value'.format(y_key))
            ax.get_xticklabels()[originalDateIndex].set_color("red")

        plt.xlabel('Date')
        plt.ylabel('Value in $')
        plt.xticks(x_ticks, rotation=45)
        plt.title('{} values for company \'{}\'\n Between dates: {} - {}\n Post publish date: {}'
                  ''.format(ListToFormattedString(columnNames),
                            self.__stockCompany,
                            self.__s_infoStartDate,
                            self.__s_infoEndDate,
                            self.__s_postOriginalDate))

        plt.legend(loc="upper left")
        plt.show()
        sys.exit()

    def plotAllSeparately(self):
        self.plotByColumnNames("Open")
        self.plotByColumnNames("High")
        self.plotByColumnNames("Low")
        self.plotByColumnNames("Close")
        self.plotByColumnNames("Volume")

    def analyzeStock(self):
        pass
