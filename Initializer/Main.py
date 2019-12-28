from Initializer import ProgramManager
from Common.Logger import Logger as Log
import Common.Constants as const


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Log()
    else:
        logger = outsourcedLogger

    # Load configurations and initialize the program
    logger.printAndLog(const.MessageType.Summarize, "Starting pre data processing...")
    programManager = ProgramManager.ProgramManager(logger)

    # Get Database
    logger.printAndLog(const.MessageType.Header, "Loading Databases files...")
    programManager.openAndPrepareRawDatabase()

    # Build local database
    logger.printAndLog(const.MessageType.Header, "Building local Databases...")
    programManager.prepareLocalDatabase()

    # Importing Stocks Databases
    logger.printAndLog(const.MessageType.Header, "Importing Stocks Databases and analyzing Stocks...")
    programManager.importStocksDatabasesForPosts()

    # Setting final train sets
    logger.printAndLog(const.MessageType.Header, "Building final Databases for learning algorithms...")
    programManager.buildFinalDatabase()

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Pre data processing finished...")

    # Debug:
    # Print database & companies (for debugging):
    # printLocalDatabase()
    # printCompaniesDict()
    # program_manager.printFailedImports()


# Run project
if __name__ == "__main__":
    main()
