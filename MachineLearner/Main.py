from datetime import datetime

import pandas as pd

import common.constants as const
from common.logger import Logger as Log
from MachineLearner.ModelTrainer import ModelTrainer
from MachineLearner.DataAnalyzer.ClassficationResultAnalyzer import ClassificationResultAnalyzer
from MachineLearner.DataAnalyzer.LinearResultAnalyzer import LinearResultAnalyzer
import MachineLearner.Classifiers as classifiers
import MachineLearner.DataFilters as dataFilters

Threshold = 5
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
    keywordsData = pd.read_csv(f'{const.databaseFolder}keywords.csv')
    keywords = keywordsData["Approved Keywords"]

    date = datetime.strptime("2020-02-15 00:00:00", const.databaseDateFormat).date()
    classificationAnalyzer = ClassificationResultAnalyzer(logger)
    classes = 15
    classification_model = ModelTrainer(logger, classes, lambda x: classifiers.classify_Nclasses(x, classes, -30, 30),
                                        f"3_Classes_Training_with_{classes}_classes", MAX_LEN,
                                        epochs, batch_size,
                                        classificationAnalyzer,
                                        lambda d: dataFilters.default_dataFilter(d))
                                        # True,
                                        # f'{const.TrainedModelDirectory}3_Classes_Training_with_threshold_5_Followers_More_Then_5000_28-03-2020_09-53-30')
    classification_model.Train(f'{const.finalDatabaseFolder}{const.trainFile}')
    classification_model.Test(f'{const.finalDatabaseFolder}{const.testFile}')

    # companies = ["Microsoft", "Google", "Intel", "Adobe", "Apple", "Amazon", "Facebook", "Twitter", "Samsung", "Activision", "Johnson"]

    # linearResultAnalyzer = LinearResultAnalyzer(logger)
    # linear_model_date = ModelTrainer(logger, 1, lambda x: classifiers.default_classifier(x),
    #                                  f"Linear_Classification_With_15_classes",
    #                                  MAX_LEN, epochs,
    #                                  batch_size,
    #                                  linearResultAnalyzer,
    #                                  lambda d: dataFilters.classify_Nclasses(d, 15, -30, 30))
    # linear_model_date.Train(f'{const.finalDatabaseFolder}{const.trainFile}')
    # linear_model_date.Test(f'{const.finalDatabaseFolder}{const.testFile}')

    #                                     lambda d: dataFilters.dSefault_dataFilter(d),
    #                                     True,
    #                                     f'{const.TrainedModelDirectory}3_Classes_Training_with_threshold_5_22-03-2020_17-24-16')
    # # classification_model.Train(f'{const.finalDatabaseFolder}{const.trainFile}')
    # classification_model.Test(f'{const.finalDatabaseFolder}{const.testFile}')

    # linearResultAnalyzer = LinearResultAnalyzer(logger)
    # linear_model = ModelTrainer(logger, 1, lambda x: classifiers.default_classifier(x),
    #                             "Linear_Classification_for_Posts_before_february_15_22-03-2020_20-35-55",
    #                             MAX_LEN, epochs,
    #                            batch_size,
    #                            linearResultAnalyzer,
    #                            lambda d: dataFilters.date_dataFilter(d, date),
    #                            True,
    #                            f'{const.TrainedModelDirectory}Linear_Classification_for_Posts_before_february_15_22-03-2020_20-35-55')
    # linear_model.Train(f'{const.finalDatabaseFolder}{const.trainFileDebug}')
    # linear_model.Test(f'{const.finalDatabaseFolder}{const.testFile}')

    # Done
    logger.printAndLog(const.MessageType.Summarize, "Learning stage finished...")


# Run project
if __name__ == "__main__":
    main()
