import ProgramManager
import NumericRepresentationService
import twitterCrawler
from logger import Logger as Log


# Project Main
def main():
    # twitterCrawler.crawl_twitter()
    run(False)


def run(process_data=False):

    Log.print_and_log("Summarize", "Welcome to our project: \"Stock prediction by twitter\" :)")
    Log.print_and_log("printLog", "")

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
    Log.print_and_log("Header", "Loading databases files...")
    program_manager.openAndPrepareRawDatabase()

    # Build local database
    Log.print_and_log("Header", "Building local databases...")
    program_manager.prepareLocalDatabase()

    # TODO: add more databases?

    # Importing stocks databases
    Log.print_and_log("Header", "Importing stocks databases and analyzing stocks...")
    program_manager.importStocksDatabasesForPosts()

    # Setting final train sets
    Log.print_and_log("Header", "Building final databases for learning algorithms")
    # TODO: remove this line
    program_manager.add_false_stocks_to_data_base()
    program_manager.build_final_database()

    # Debug:
    program_manager.printFailedImports()


# Run project
main()
