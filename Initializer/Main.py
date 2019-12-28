from Initializer import ProgramManager
from Common.Logger import Logger as Log
import Common.Constants as c


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Log()
    else:
        logger = outsourcedLogger

    # Load configurations and initialize the program
    programManager = ProgramManager.ProgramManager(logger)

    # Get Database
    logger.printAndLog(c.MessageType.Header, "Loading Databases files...")
    programManager.openAndPrepareRawDatabase()

    # Build local database
    logger.printAndLog(c.MessageType.Header, "Building local Databases...")
    programManager.prepareLocalDatabase()

    # TODO: add more Databases?

    # Importing Stocks Databases
    logger.printAndLog(c.MessageType.Header, "Importing Stocks Databases and analyzing Stocks...")
    programManager.importStocksDatabasesForPosts()

    # Setting final train sets
    logger.printAndLog(c.MessageType.Header, "Building final Databases for learning algorithms")
    # TODO: remove this line
    programManager.add_false_stocks_to_data_base()
    programManager.build_final_database()

    # Debug:
    # Print database & companies (for debugging):
    # printLocalDatabase()
    # printCompaniesDict()
    # program_manager.printFailedImports()

    # NLP database process
    # numeric_representation_service = NumericRepresentationService.NumericRepresentationService()
    # train_iterator, test_iterator = numeric_representation_service.get_numeric_representation_of_final_data()


# Run project
if __name__ == "__main__":
    main()
