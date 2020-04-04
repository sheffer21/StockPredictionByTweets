from abc import ABC
import MachineLearner.DataAnalyzer.StatisticsProvider as stat


class DataAnalyzer(ABC):

    def __init__(self, logger):
        self.logger = logger
        self.nb_eval_steps = 0

    # Database Analyzer---------------------------------------------------------
    def AnalyzeDataSet(self, sentences, labels, followers, runName):
        stat.Plot_Distributions(followers, runName)
        pass

    # Validation Analyzer---------------------------------------------------------
    def StartValidation(self):
        self.nb_eval_steps = 0
        pass

    def PerformValidationStep(self, logits, label_ids):
        # Track the number of batches
        self.nb_eval_steps += 1
        pass

    def FinishValidation(self):
        pass

    # Test Analyzer---------------------------------------------------------
    def PrintTestResult(self, true_labels, predictions, runName):
        pass
