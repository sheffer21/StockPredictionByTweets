import csv
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

    def __init__(self, stockCompany, stockSymbol, infoStartDate, infoEndDate, rawDataPath):
        self.__stockCompany = stockCompany
        self.__s_stockSymbol = stockSymbol
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
        for rowDate in self.__s_stockData:
            x.append(rowDate)
            y.append(float(self.__s_stockData[rowDate][StockInfo.columnIdx.get(columnName)]))

        plt.plot(x, y, color='red')
        plt.xlabel('Date')
        plt.ylabel('Value by date')
        plt.xticks(x, rotation=45)
        plt.title('Values of \'{}\' for company {}\n Between dates: {} - {}'
                  ''.format(columnName,
                            self.__stockCompany,
                            self.__s_infoStartDate,
                            self.__s_infoEndDate))
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
