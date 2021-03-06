from MachineLearner.DataAnalyzer import DataAnalyzer
import numpy as np
import common.constants as const
from sklearn.metrics import matthews_corrcoef
import MachineLearner.DataAnalyzer.StatisticsProvider as stat


class ClassificationResultAnalyzer(DataAnalyzer.DataAnalyzer):

    def __init__(self, logger, classes):
        super().__init__(logger, DataAnalyzer.AnalyzerType.Classification, classes)
        self.eval_accuracy = 0

    # Database Analyzer---------------------------------------------------------
    def AnalyzeDataSet(self, sentences, labels, followers, runName):
        super().AnalyzeDataSet(sentences, labels, followers, runName)
        stat.Plot_DataBase_Labels_Statistics(labels, ["Neutral", "Negative", "Positive"], runName)

    # Validation Analyzer---------------------------------------------------------
    def StartValidation(self):
        super().StartValidation()
        self.eval_accuracy = 0

    def PerformValidationStep(self, logits, label_ids):
        super().PerformValidationStep(logits, label_ids)
        # Calculate the accuracy for this batch of test sentences.
        self.eval_accuracy += self.flat_accuracy(logits, label_ids)

    def FinishValidation(self):
        super().FinishValidation()
        self.logger.printAndLog(const.MessageType.Regular,
                                "   Accuracy: {0:.2f}".format(self.eval_accuracy / self.nb_eval_steps))

    # Test Analyzer---------------------------------------------------------
    def PrintTestResult(self, true_labels, predictions, companies, dates, runName):
        self.PrintResults([item for sublist in true_labels for item in sublist],
                          [item for sublist in predictions for item in sublist],
                          companies,
                          dates,
                          runName)
        self.Get_MCC(true_labels, predictions, runName)

    def GetBatchPredictions(self, true_labels, predictions, companies, dates, runName):
        true_labels_flat, predictions_flat = self.GetFlattenVectors(true_labels, predictions)
        return super().GetBatchPredictions(true_labels_flat, predictions_flat, companies, dates, runName)

    # Function to calculate the accuracy of our predictions vs labels
    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def Get_MCC(self, true_labels, predictions, runName):
        matthews_set = []

        # Evaluate each test batch using Matthew's correlation coefficient
        self.logger.printAndLog(const.MessageType.Regular, 'Calculating Matthews Corr. Coef. for each batch...')

        # For each input batch...
        for i in range(len(true_labels)):
            # The predictions for this batch are a 2-column ndarray (one column for "0"
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

            # Calculate and store the coef for this batch.
            matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
            matthews_set.append(matthews)

        # Combine the predictions for each batch into a single list of 0s and 1s.
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        predictions_differences = self.difference(flat_true_labels, flat_predictions)
        total = len(flat_true_labels)
        correct_predictions = len(flat_true_labels) - len(predictions_differences)
        self.logger.printAndLog(const.MessageType.Regular,
                                f"Positive samples: {correct_predictions} of {total} "
                                f"({(correct_predictions / total * 100.0)})")

        # Calculate the MCC
        mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
        self.logger.printAndLog(const.MessageType.Regular, 'MCC: %.3f' % mcc)

    @staticmethod
    def difference(list1, list2):
        list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
        return list_dif

    def GetFlattenVectors(self, true_labels, predictions):
        # Combine the predictions for each batch into a single list of 0s and 1s.
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_predictions = [item for item in flat_predictions]

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        return flat_true_labels, flat_predictions
