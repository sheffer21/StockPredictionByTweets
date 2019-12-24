import twitterCrawler
from mergeService import merge


def main():
    twitterCrawler.crawl_twitter()
    merge()


main()
