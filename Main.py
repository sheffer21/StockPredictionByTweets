import ProgramManager


# Project Main
def main():
    # Load configurations and initialize the program
    programManager = ProgramManager.ProgramManager()
    programManager.printAndLog("Summarize", "Welcome to our project: \"Stock prediction by twitter\" :)")
    programManager.printAndLog("printLog", "")

    # Get Database
    programManager.printAndLog("Header", "Loading databases files...")
    programManager.openAndPrepareRawDatabase()

    # Build local database
    programManager.printAndLog("Header", "Building local databases...")
    programManager.prepareLocalDatabase()

    # TODO: add more databases?

    # Importing stocks databases
    programManager.printAndLog("Header", "Importing stocks databases and analyzing stocks...")
    programManager.importStocksDatabasesForPosts()

    # Setting final train sets
    programManager.printAndLog("Header", "Building final databases for learning algorithms")
    programManager.buildFinalDatabase()

    # Debug:
    programManager.printFailedImports()

    return

    for row in programManager.database:
        print(row)

    # Print database & companies (for debugging):
    # printLocalDatabase()
    # printCompaniesDict()


# Run project
main()
