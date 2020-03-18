from common import logger as Logger
from common import constants as const
from TwitterCrawler import TwitterCrawler
from TwitterCrawler.DataBaseOperationsService import DataBaseOperationsService
from TwitterCrawler.DataBaseStatistics import DataBaseStatistics


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
