import common.constants as const
from common.logger import Logger as Log
from MachineLearner.ModelTrainer import ModelTrainer
# from MachineLearner.ResultAnalyzer.ClassficationResultAnalyzer import ClassificationResultAnalyzer
from MachineLearner.ResultAnalyzer.LinearResultAnalyzer import LinearResultAnalyzer

PositiveThreshold = 0
NegativeThreshold = 0
MAX_LEN = 64
# Number of training epochs (authors recommend between 2 and 4)
epochs = 4
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.
batch_size = 16


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
    # model = ModelTrainer(logger, 3, lambda x: classify_3classes(x), "3_Classes_Training", MAX_LEN, epochs, batch_size)
    # classificationAnalyzer = ClassificationResultAnalyzer()
    linearResultAnalyzer = LinearResultAnalyzer(logger)
    model = ModelTrainer(logger, 1, lambda x: classify_linear(x), "Linear_Classification", MAX_LEN, epochs, batch_size,
                         linearResultAnalyzer)
    model.Train(f'{const.finalDatabaseFolder}{const.trainFileDebug}')
    model.Test(f'{const.finalDatabaseFolder}{const.testFileDebug}')

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
