class StockInfo:
    __m_totalInstances = 0

    def __init__(self, stockSymbol, infoStartDate, infoEndDate, rawData):
        self.__s_stockSymbol = stockSymbol
        self.__s_infoStartDate = infoStartDate
        self.__s_infoEndDate = infoEndDate
        self.__s_dataPath = ""

        StockInfo.__m_totalInstances += 1
