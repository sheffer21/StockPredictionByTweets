from common import logger as Logger
from common import constants as const
import TwitterCrawler.main as tc
import Preprocessor.Main as init
import MachineLearner.Main as learner


def main():
    logger = Logger.Logger()

    logger.printAndLog(const.MessageType.Summarize, "Welcome to our project: \"Stock prediction by twitter\" :)")
    logger.printAndLog(const.MessageType.printLog, "")

    # tweetsCrawler = input(const.MAIN_HEADER + "Do you want to run the twitter crawler? \n"
    #                                          "Twitter crawler uses twitter's REST API "
    #                                          "to fetch new tweets.\n"
    #                                          "Type <yes/no>: " + const.MAIN_ENDC)

    # if tweetsCrawler == "yes":
    #     tc.main(logger)
    # else:
    #     logger.printAndLog(const.MessageType.Summarize, "TweetsCrawler not ran")

    # runPreProcessing = input(const.MAIN_HEADER + "Do you want to run the Preprocessor? \n"
    #                                              "Preprocessor uses fetched databases to build and analyze "
    #                                              "stocks database prior the learning stage. \n"
    #                                              "Type <yes/no>: " + const.MAIN_ENDC)

    # if runPreProcessing == "yes":
    #     init.main(logger)
    # else:
    #     logger.printAndLog(const.MessageType.Summarize, "Preprocessor not ran")

    # runMachineLearning = input(const.MAIN_HEADER + "Do you want to run the machine learner? \n"
    #                                                "description \n"
    #                                                "Type <yes/no>: " + const.MAIN_ENDC)

    # if runMachineLearning == "yes":
    learner.main(logger)
    # else:
    #    logger.printAndLog(const.MessageType.Summarize, "MachineLearner not ran")

    # runAfterMath = input("Do you want to run after math? \n"
    #                      "After math lets you run operations like looking in graphs and getting statistics"
    #                      "from the stocks data and learning stages."
    #                      "Type <yes/no>: " + const.ENDC)


# Run project
main()
