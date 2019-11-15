import xlrd
from Company import Company
from Post import Post


# Project methods:
def openAndPrepareDatabase(path, sheetName):
    wb = xlrd.open_workbook(path)
    database = wb.sheet_by_name(sheetName)

    return database


def prepareLocalDatabase(database):
    for databaseRowIndex in range(1, database.nrows):
        postId = databaseRowIndex
        postText = database.cell_value(databaseRowIndex, POST_TEXT_COLUMN)
        postTimestamp = database.cell_value(databaseRowIndex, POST_TIMESTAMP_COLUMN)
        postSource = database.cell_value(databaseRowIndex, POST_SOURCE_COLUMN)
        postSymbols = database.cell_value(databaseRowIndex, POST_SYMBOLS_COLUMN)
        postCompany = database.cell_value(databaseRowIndex, POST_COMPANY_COLUMN)
        postUrl = database.cell_value(databaseRowIndex, POST_URL_COLUMN)
        postVerified = database.cell_value(databaseRowIndex, POST_VERIFIED_COLUMN)

        postCompaniesList = []
        postSymbolsParsed = postSymbols.split('-')
        postCompaniesParsed = postCompany.split('*')

        for companiesArrayIndex in range(len(postSymbolsParsed)):
            newCompany = Company(postSymbolsParsed[companiesArrayIndex], postCompaniesParsed[companiesArrayIndex])
            postCompaniesList.append(newCompany)
            companiesDict[postSymbolsParsed[companiesArrayIndex]] = postCompaniesParsed[companiesArrayIndex]

        newPost = Post(postId, postText, postTimestamp, postSource, postCompaniesList,
                       postUrl, postVerified, "unknown", "unknown")

        postsList.append(newPost)


def printLocalDatabase(maxIterations):
    iterationsCount = 0  # For debugging
    for post in postsList:
        print(post.description)
        iterationsCount += 1

        # For debugging
        if iterationsCount == maxIterations:
            break


def printCompaniesDict(maxIterations):
    iterationsCount = 0  # For debugging
    for companySymbol in companiesDict:
        print("Company symbol: {}, company name: {}".format(companySymbol, companiesDict[companySymbol]))
        iterationsCount += 1

        # For debugging
        if iterationsCount == maxIterations:
            break


# Project properties:
postsList = []
companiesDict = {}
databasePath = "databases/stockerbot-export.xlsx"
workSheetName = "stockerbot-export"
databaseFileName = "stockerbot-export.xlsx"
POST_ID_COLUMN = 0
POST_TEXT_COLUMN = 1
POST_TIMESTAMP_COLUMN = 2
POST_SOURCE_COLUMN = 3
POST_SYMBOLS_COLUMN = 4
POST_COMPANY_COLUMN = 5
POST_URL_COLUMN = 6
POST_VERIFIED_COLUMN = 7
printPostsLimit = 10  # For debugging
printCompaniesLimit = 10  # For debugging


# Project Main
def main():
    print("Welcome to our project :)")
    print("Loading databases files...")

    # Get Database
    database = openAndPrepareDatabase(databasePath, workSheetName)
    # Build local database
    prepareLocalDatabase(database)

    # TODO: Get stocks databases for each company
    # TODO: Analyze stocks databases
    # TODO: Initial database process
    # TODO: NLP database process

    # Print database & companies (for debugging)
    printLocalDatabase(printPostsLimit)
    printCompaniesDict(printCompaniesLimit)


# Run project
main()
