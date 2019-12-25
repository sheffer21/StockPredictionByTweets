import ProgramManager
import NumericRepresentationService
from common.logger import Logger as Log
import common.constants as c


# Project Main
def main(process_data=False):

    Log.print_and_log(c.MessageType.Summarize.value, "Welcome to our project: \"Stock prediction by twitter\" :)")
    Log.print_and_log(c.MessageType.printLog.value, "")

    if process_data:
        prepare_database()

    # NLP database process
    numeric_representation_service = NumericRepresentationService.NumericRepresentationService()
    train_iterator, test_iterator = numeric_representation_service.get_numeric_representation_of_final_data()

    # TODO: Analyze stocks databases

    # Print database & companies (for debugging):
    # printLocalDatabase()
    # printCompaniesDict()


def prepare_database():

    # Load configurations and initialize the program
    program_manager = ProgramManager.ProgramManager()

    # Get Database
    Log.print_and_log(c.MessageType.Header.value, "Loading databases files...")
    program_manager.openAndPrepareRawDatabase()

    # Build local database
    Log.print_and_log(c.MessageType.Header.value, "Building local databases...")
    program_manager.prepareLocalDatabase()

    # TODO: add more databases?

    # Importing stocks databases
    Log.print_and_log(c.MessageType.Header.value, "Importing stocks databases and analyzing stocks...")
    program_manager.importStocksDatabasesForPosts()

    # Setting final train sets
    Log.print_and_log(c.MessageType.Header.value, "Building final databases for learning algorithms")
    # TODO: remove this line
    program_manager.add_false_stocks_to_data_base()
    program_manager.build_final_database()

    # Debug:
    # Print database & companies (for debugging):
    # printLocalDatabase()
    # printCompaniesDict()
    # program_manager.printFailedImports()


# Run project
main()
