import common.constants as const
from common.logger import Logger as Log
from MachineLearner.ModelTrainer import ModelTrainer
from MachineLearner.ResultAnalyzer.ClassficationResultAnalyzer import ClassificationResultAnalyzer
from MachineLearner.ResultAnalyzer.LinearResultAnalyzer import LinearResultAnalyzer

PositiveThreshold = 1
NegativeThreshold = 1
MAX_LEN = 64
# Number of training epochs (authors recommend between 2 and 4)
epochs = 4
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.
batch_size = 32


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
    classificationAnalyzer = ClassificationResultAnalyzer(logger)
    classification_model = ModelTrainer(logger, 3, lambda x: classify_3classes(x), "3_Classes_Training", MAX_LEN,
                                        epochs, batch_size,
                                        classificationAnalyzer)
    classification_model.Train(f'{const.finalDatabaseFolder}{const.trainFile}')
    classification_model.Test(f'{const.finalDatabaseFolder}{const.testFile}')

    linearResultAnalyzer = LinearResultAnalyzer(logger)
    linear_model = ModelTrainer(logger, 1, lambda x: classify_linear(x), "Linear_Classification", MAX_LEN, epochs,
                                batch_size,
                                linearResultAnalyzer)
    linear_model.Train(f'{const.finalDatabaseFolder}{const.trainFile}')
    linear_model.Test(f'{const.finalDatabaseFolder}{const.testFile}')

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Learning stage finished...")


# Run project
if __name__ == "__main__":
    main()


def classify_3classes(label):
    if label > PositiveThreshold:
        return 2
    if label < -NegativeThreshold:
        return 1
    return 0


def classify_linear(label):
    return label
