import ProgramManager
import NumericRepresentationService
from Database import Database


# Project Main
def main(process_data=False):

    # Load configurations and initialize the program
    program_manager = ProgramManager.ProgramManager()
    program_manager.printAndLog("Summarize", "Welcome to our project: \"Stock prediction by twitter\" :)")
    program_manager.printAndLog("printLog", "")

    if process_data:
        prepare_database(program_manager)

    # NLP database process
    numeric_representation_service = NumericRepresentationService.NumericRepresentationService(program_manager)
    train_iterator, test_iterator = numeric_representation_service.get_numeric_representation_of_final_data()
    train_iterator
    test_iterator

    # TODO: Analyze stocks databases

    # Print database & companies (for debugging):
    # printLocalDatabase()
    # printCompaniesDict()


def prepare_database(program_manager):

    # Get Database
    program_manager.printAndLog("Header", "Loading databases files...")
    program_manager.openAndPrepareRawDatabase()

    # Build local database
    program_manager.printAndLog("Header", "Building local databases...")
    program_manager.prepareLocalDatabase()

    # TODO: add more databases?

    # Importing stocks databases
    program_manager.printAndLog("Header", "Importing stocks databases and analyzing stocks...")
    program_manager.importStocksDatabasesForPosts()

    # Setting final train sets
    program_manager.printAndLog("Header", "Building final databases for learning algorithms")
    # TODO: remove this line
    program_manager.add_false_stocks_to_data_base()
    program_manager.build_final_database()

    # Debug:
    program_manager.printFailedImports()


# Run project
main()
