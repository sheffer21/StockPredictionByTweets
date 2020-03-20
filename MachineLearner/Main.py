import common.constants as const
from common.logger import Logger as Log
from MachineLearner.ModelTrainer import ModelTrainer

PositiveThreshold = 1
NegativeThreshold = 1


def main(outSourcedLogger=None):

    if outSourcedLogger is None:
        logger = Log()
    else:
        logger = outSourcedLogger

    logger.printAndLog(const.MessageType.Summarize, "Starting machine learning algorithms...")

    # Perform word embedding
    # numericRepresentationService = NumericRepresentationService(logger)
    # train_iterator, test_iterator, vocab_size, prediction_vocab_size \
    #    = numericRepresentationService.getNumericRepresentationOfFinalData()

    # Train the model
    model = ModelTrainer(logger, 3, lambda x: classify(x))
    model.Train(f'{const.finalDatabaseFolder}{const.trainFileDebug}')
    model.Test(f'{const.finalDatabaseFolder}{const.testFileDebug}')

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Learning stage finished...")


# Run project
if __name__ == "__main__":
    main()


def classify(label):
    if label > PositiveThreshold:
        return 2
    if label < -NegativeThreshold:
        return 1
    return 0
