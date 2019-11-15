from Company import Company


class Post:
    __m_totalPosts = 0
    _test = 50

    def __init__(self, postId, text, date, source, companies, url, verified, stock_delta, stock_change):
        self.__p_id = postId
        self.__p_text = text
        self.__p_date = date
        self.__p_source = source
        self.__p_companies = companies
        self.__p_url = url
        self.__p_verified = verified
        self.__p_stock_delta = stock_delta
        self.__p_stock_change = stock_change
        self.__p_total_impact = stock_change

        Post.__m_totalPosts += 1

    @classmethod
    def totalPosts(cls):
        return cls.__m_totalPosts

    @property
    def description(self):
        postDescription = "Post id: {},\n" \
                          "\t text: {},\n" \
                          "\t date: {},\n" \
                          "\t source: {},\n" \
                          "\t url: {},\n" \
                          "\t verified: {},\n" \
                          "\t stock delta: {},\n" \
                          "\t stock change: {},\n" \
                          "\t total impact: {},\n" \
                          "".format(self.__p_id, self.__p_text, self.__p_date, self.__p_source, self.__p_url,
                                    self.__p_verified, self.__p_stock_delta, self.__p_stock_change, self.__p_total_impact)

        companiesDescription = "This post is associated with the following companies:\n"

        for company in self.__p_companies:
            companiesDescription = companiesDescription + "\t" + company.description + "\n"

        return postDescription + companiesDescription

    @property
    def totalImpact(self):
        return self.__p_total_impact

    @property
    def date(self):
        return self.__p_date

    @property
    def text(self):
        return self.__p_text

    def addCompany(self, company):
        self.__p_companies.append(company)

    def addCompanies(self, companiesList):
        self.__p_companies.extend(companiesList)

    def setPostStockDelta(self, stock_delta):
        self.__p_stock_delta = stock_delta

    def setPostStockDelta(self, stock_change):
        self.__p_stock_change = stock_change

    def setPostStockDelta(self, total_impact):
        self.__p_total_impact = total_impact

