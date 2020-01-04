import constants as const
from logger import Logger as Log
import NumericRepresentationService


def main(outSourcedLogger=None):

    if outSourcedLogger is None:
        logger = Log()
    else:
        logger = outSourcedLogger

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