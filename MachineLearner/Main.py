import Common.Constants as const
from Common.Logger import Logger as Log
from MachineLearner import NumericRepresentationService


def main(outsourcedLogger=None):

    if outsourcedLogger is None:
        logger = Log()
    else:
        logger = outsourcedLogger

    logger.printAndLog(const.MessageType.Summarize, "Starting machine learning algorithms...")

    # NLP database process
    logger.printAndLog(const.MessageType.Header, "Exporting data to numeric representation...")
    numeric_representation_service = NumericRepresentationService.NumericRepresentationService(logger)
    train_iterator, test_iterator = numeric_representation_service.get_numeric_representation_of_final_data()

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Learning stage finished...")


# Run project
if __name__ == "__main__":
    main()