import Common.Constants as const
from Common.Logger import Logger as Log
from MachineLearner import NumericRepresentationService


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Log()
    else:
        logger = outsourcedLogger

    logger.printAndLog(const.MessageType.Summarize, "Starting machine learning algorithms...")
    numericRepresentationService = NumericRepresentationService.NumericRepresentationService(logger)

    # NLP database process
    logger.printAndLog(const.MessageType.Header, "Exporting data to numeric representation...")
    train_iterator, test_iterator = numericRepresentationService.getNumericRepresentationOfFinalData()

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Learning stage finished...")


# Run project
if __name__ == "__main__":
    main()