from pandas import np

from MachineLearner.DataAnalyzer import DataAnalyzer
import common.constants as const
from math import sqrt
from sklearn.metrics import mean_squared_error


class LinearResultAnalyzer(DataAnalyzer.DataAnalyzer):

    def __init__(self, logger):
        super().__init__(logger)
        self.eval_meanSquare, self.coefficient = 0, 0

    # Database Analyzer---------------------------------------------------------
    def AnalyzeDataSet(self, sentences, labels, followers, runName):
        super().AnalyzeDataSet(sentences, labels, followers, runName)

    # Validation Analyzer---------------------------------------------------------
    def StartValidation(self):
        super().StartValidation()
        self.eval_meanSquare = 0
        self.coefficient = 0

    def PerformValidationStep(self, logits, label_ids):
        super().PerformValidationStep(logits, label_ids)

        self.eval_meanSquare += self.GetMeanSquare(logits, label_ids)
        self.coefficient += self.correlCo(logits, label_ids)

    def FinishValidation(self):
        super().FinishValidation()

        self.logger.printAndLog(const.MessageType.Regular,
                                f"  Coefficient: {self.coefficient / self.nb_eval_steps:.2f}")
        self.logger.printAndLog(const.MessageType.Regular,
                                f"  Mean Square: {self.eval_meanSquare / self.nb_eval_steps:.2f}")

    # Test Analyzer---------------------------------------------------------
    def PrintTestResult(self, true_labels, predictions, runName):
        np.savetxt(f'{const.TrainedModelDirectory}{runName}/test_result.out', (true_labels, predictions),
                   delimiter=',')

        self.logger.printAndLog(const.MessageType.Regular,
                                f'   Coefficient: {self.correlCo(true_labels, predictions):.2f}')
        self.logger.printAndLog(const.MessageType.Regular,
                                f'   Mean Square: {self.GetMeanSquare(true_labels, predictions):.2f}')

    @staticmethod
    def GetMeanSquare(y_actual, y_predicted):
        return sqrt(mean_squared_error(y_actual, y_predicted))

    @staticmethod
    def mean(someList):
        total = 0
        for a in someList:
            total += float(a)
        mean = total / len(someList)
        return mean

    def standDev(self, someList):
        listMean = self.mean(someList)
        dev = 0.0
        for i in range(len(someList)):
            dev += (someList[i] - listMean) ** 2
        dev = dev ** (1 / 2.0)
        return dev

    def correlCo(self, someList1, someList2):
        # First establish the means and standard deviations for both lists.
        xMean = self.mean(someList1)
        yMean = self.mean(someList2)
        xStandDev = self.standDev(someList1)
        yStandDev = self.standDev(someList2)
        # r numerator
        rNum = 0.0
        for i in range(len(someList1)):
            rNum += (someList1[i] - xMean) * (someList2[i] - yMean)

        # r denominator
        rDen = xStandDev * yStandDev

        r = rNum / rDen
        return r
