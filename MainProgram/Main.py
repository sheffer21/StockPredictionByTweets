from Common import Logger
import Common.Constants as const
import TwitterCrawler.Main as tc
import Initializer.Main as init


def main():
    logger = Logger.Logger()

    logger.printAndLog(const.MessageType.Summarize.value, "Welcome to our project: \"Stock prediction by twitter\" :)")
    logger.printAndLog(const.MessageType.printLog.value, "")

    # TODO : add colors

    tweetsCrawler = input("Do you want to run the twitter crawler? \n"
                          "Twitter crawler uses twitter's REST API to fetch new tweets.\n"
                          "Type <yes/no>: ")

    if tweetsCrawler == "yes":
        logger.printAndLog(const.MessageType.Summarize.Regular, "Running TweetsCrawler")
        tc.main(logger)
        logger.printAndLog(const.MessageType.Summarize.Regular, "TweetsCrawler finished")
    else:
        logger.printAndLog(const.MessageType.Summarize.Regular, "TweetsCrawler not ran")

    runInitializer = input("Do you want to run the Initializer? \n"
                           "Initializer uses fetched databases to build and analyze "
                           "stocks database prior the learning stage. \n"
                           "Type <yes/no>: ")

    if runInitializer == "yes":
        logger.printAndLog(const.MessageType.Summarize.Regular, "Running Initializer")
        init.main(logger)
        logger.printAndLog(const.MessageType.Summarize.Regular, "Initializer finished")
    else:
        logger.printAndLog(const.MessageType.Summarize.Regular, "Initializer not ran")

    # runMachineLearning = input("Do you want to run the machine learner? \n"
    #                            "description \n"
    #                            "Type <yes/no>: ")
    #
    # if runMachineLearning == "yes":
    #     logger.printAndLog(const.MessageType.Summarize.Regular, "Running MachineLearner")
    #     # MachineLearnerMain.main(logger)
    #     logger.printAndLog(const.MessageType.Summarize.Regular, "MachineLearner finished")
    # else:
    #     logger.printAndLog(const.MessageType.Summarize.Regular, "MachineLearner not ran")
    #
    # runAfterMath = input("Do you want to run after math? \n"
    #                      "After math lets you run operations like looking in graphs and getting statistics"
    #                      "from the stocks data and learning stages."
    #                      "Type <yes/no>: ")
    #
    # if runAfterMath == "yes":
    #     logger.printAndLog(const.MessageType.Summarize.Regular, "Running after math")
    #     # Initializer.Main.main(logger)
    #     logger.printAndLog(const.MessageType.Summarize.Regular, "After mat finished")
    # else:
    #     logger.printAndLog(const.MessageType.Summarize.Regular, "After math not ran")
    #
    # logger.printAndLog(const.MessageType.printLog.value, "")
    # logger.printAndLog(const.MessageType.Summarize.value, "Program finished.")


# Run project
main()
