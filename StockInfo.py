import csv
import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys


class StockInfo:
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

        self.parseData()
        self.analyzeStock()
        self.plotByColumnName("Close")

    def parseData(self):
        with open(self.__s_dataPath, newline='') as dataFile:
            reader = csv.DictReader(dataFile)
            for row in reader:
                rowKey = row['Date']
                rowValue = [
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    row['Dividends'],
                    row['Stock Splits']
                ]

                self.__s_stockData[rowKey] = rowValue

    def analyzeStock(self):
        pass

    def plotByColumnName(self, columnName):
        x = []
        y = []
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
            y.append(float(self.__s_stockData[rowDate][StockInfo.columnIdx.get(columnName)]))

        fig, ax = plt.subplots()
        ax.plot(x, y, color='red')
        ax.get_xticklabels()[originalDateIndex].set_color("red")
        plt.xlabel('Date')
        plt.ylabel('Value in $')
        plt.xticks(x_ticks, rotation=45)
        plt.title('Values of \'{}\' for company {}\n Between dates: {} - {}\n Post publish date: {}'
                  ''.format(columnName,
                            self.__stockCompany,
                            self.__s_infoStartDate,
                            self.__s_infoEndDate,
                            self.__s_postOriginalDate))

        valueLegend = mpatches.Patch(color='red', label='Value by date')
        plt.legend(handles=[valueLegend])
        plt.grid()
        plt.show()
        sys.exit()

    def plotAll(self):
        self.plotByColumnName("Open")
        self.plotByColumnName("High")
        self.plotByColumnName("Low")
        self.plotByColumnName("Close")
        self.plotByColumnName("Volume")
        self.plotByColumnName("Dividends")
        self.plotByColumnName("Stock Splits")
