from Common.Logger import Logger as Log
from TwitterCrawler import MergeService, Twitter


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Log()
    else:
        logger = outsourcedLogger

    Twitter.TwitterCrawler.crawlTwitter(logger)
    MergeService.MergeService.merge(logger)


if __name__ == "__main__":
    main()
