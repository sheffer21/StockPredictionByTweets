import common.constants as const
from common.logger import Logger as Log
from MachineLearner.NumericRepresentationService import NumericRepresentationService
from MachineLearner.ModelTrainer import ModelTrainer


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
    model = ModelTrainer(logger)
    model.Train()
    model.Test()

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Learning stage finished...")


# Run project
if __name__ == "__main__":
    main()
