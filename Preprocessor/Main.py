import Preprocessor.Manager
from common import logger as Logger
from common import constants as const


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Logger()
    else:
        logger = outsourcedLogger

    # Load configurations and initialize the program
    logger.printAndLog(const.MessageType.Summarize, "Starting pre data processing...")
    preProcessor = Manager.PreProcessor(logger)

    # Get Database
    logger.printAndLog(const.MessageType.Header, "Loading Databases files...")
    preProcessor.openAndPrepareRawDatabase()

    # Build local database
    logger.printAndLog(const.MessageType.Header, "Building local Databases...")
    preProcessor.prepareLocalDatabase()

    # Importing Stocks Databases
    logger.printAndLog(const.MessageType.Header, "Importing Stocks Databases and analyzing Stocks...")
    preProcessor.importStocksDatabasesForPosts()

    # Setting final train sets
    logger.printAndLog(const.MessageType.Header, "Building final Databases for learning algorithms...")
    preProcessor.buildFinalDatabase()

    # Print and export failed stock imports
    logger.printAndLog(const.MessageType.Header, "Listing failed imports...")
    preProcessor.printAndExportFailedImports()

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Pre data processing finished...")


# Run project
if __name__ == "__main__":
    main()
