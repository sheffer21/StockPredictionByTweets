# define the Vehicle class
class Company:

    def __init__(self, symbol, name):
        self.__c_symbol = symbol
        self.__c_name = name

    @property
    def stockSymbol(self):
        return self.__c_symbol

    @property
    def name(self):
        return self.__c_name

    @property
    def description(self):
        desc_str = "Company name: %s, company symbol: %s" % (self.__c_symbol, self.__c_name)
        return desc_str
