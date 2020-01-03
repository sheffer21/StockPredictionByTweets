import logger as Log
import constants as const
from TwitterCrawler import TwitterCrawler
from DataBaseOperationsService import DataBaseOperationsService
from DataBaseStatistics import DataBaseStatistics


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Log.Logger()
    else:
        logger = outsourcedLogger

    logger.printAndLog(const.MessageType.Summarize.value, "Gathering data...")
    twitterService = TwitterCrawler(logger)

    # Fetch tweets
    logger.printAndLog(const.MessageType.Header.value, "Collecting data from twitter...")
    twitterService.crawlTwitter()

    # Merge files
    operations = DataBaseOperationsService(logger)
    logger.printAndLog(const.MessageType.Header.value, "Merging new data with current database...")
    operations.merge()

    # Plot statistics
    statistics = DataBaseStatistics(logger)
    statistics.PublishDataBaseCompaniesGraph()
    statistics.PublishDataBaseCompaniesKeywordsGraph()

    # Done
    logger.printAndLog(const.MessageType.Summarize.value, "Gathering finished...")


if __name__ == "__main__":
    main()
