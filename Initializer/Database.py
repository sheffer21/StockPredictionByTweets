
class Database:

    def __init__(self):
        self.__d_x = []
        self.__d_y = []

    def addItem(self, x, y):
        self.__d_x.append(x)
        self.__d_y.append(y)
