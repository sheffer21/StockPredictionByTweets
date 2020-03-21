from abc import ABC


class ResultAnalyzer:

    def __init__(self, logger):
        self.logger = logger
        self.nb_eval_steps = 0

    def StartValidation(self):
        self.nb_eval_steps = 0
        pass

    def PerformValidationStep(self, logits, label_ids):
        # Track the number of batches
        self.nb_eval_steps += 1
        pass

    def FinishValidation(self):
        pass

    def PrintTestResult(self, true_labels, predictions):
        pass
