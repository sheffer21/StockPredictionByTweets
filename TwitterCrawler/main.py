from Common.Logger import Logger as Log
import Common.Constants as const
from TwitterCrawler import TwitterCrawler


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Log()
    else:
        logger = outsourcedLogger

    logger.printAndLog(const.MessageType.Summarize, "Gathering data...")
    twitterService = TwitterCrawler.TwitterCrawler(logger)

    # Fetch tweets
    logger.printAndLog(const.MessageType.Header, "Collecting data from twitter...")
    twitterService.crawlTwitter()

    # Merge files
    logger.printAndLog(const.MessageType.Header, "Merging new data with current database...")
    twitterService.merge()

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Gathering finished...")


if __name__ == "__main__":
    main()
