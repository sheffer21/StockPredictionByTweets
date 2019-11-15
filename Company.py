# define the Vehicle class
class Company:
    __m_totalCompanies = 0

    def __init__(self, symbol, name):
        self.__m_symbol = symbol
        self.__m_name = name

    @property
    def description(self):
        desc_str = "Company name: %s, company symbol: %s" % (self.__m_symbol, self.__m_name)

        return desc_str
