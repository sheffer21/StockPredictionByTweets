from abc import ABC
import datetime
import pandas as pd
import common.constants as const
import MachineLearner.DataAnalyzer.StatisticsProvider as stat
import numpy as np
import matplotlib.pyplot as plt
import enum


class DataAnalyzer(ABC):

    def __init__(self, logger, analyzerType, classes):
        self.logger = logger
        self.nb_eval_steps = 0
        self.analyzerType = analyzerType
        self.classes = classes

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
    def PrintTestResult(self, true_labels, predictions, companies, dates, runName):
        pass

    def GetBatchPredictions(self, true_labels, predictions, companies, dates, runName):
        batch_trueLabels, batch_predictions, batch_company, batch_date = [], [], [], []

        data = pd.concat([pd.DataFrame([
            [datetime.datetime.strptime(dates[index], const.databaseDateFormat).date(),
             companies[index],
             predictions[index],
             true_labels[index]]],
            columns=['date', 'company', 'prediction', 'true_label'])
            for index in range(len(predictions))])

        grouped_data_by_date_and_company = data.groupby(['date', 'company'])

        index = 0

        if self.analyzerType == AnalyzerType.Linear:
            bins_distribution = 10
        else:
            d = 1
            left_of_first_bin = 0 - float(d) / 2
            right_of_last_bin = self.classes - 1 + float(d) / 2
            bins_distribution = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        for group_name, df_group in grouped_data_by_date_and_company:
            index += 1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            predictions = df_group['prediction']

            n, bins, patches = ax.hist(predictions, bins=bins_distribution,
                                       density=True, fc='k', alpha=0.3)

            chosen_bin = np.argmax(n)
            if self.analyzerType == AnalyzerType.Classification:
                prediction = chosen_bin
            else:
                prediction = bins[chosen_bin]

            for i, item in df_group.iterrows():
                batch_trueLabels.append(item['true_label'])
                batch_predictions.append(prediction)
                batch_company.append(item['company'])
                batch_date.append(item['date'])

            # plt.show()
            plt.close()

        self.PrintResults(batch_trueLabels, batch_predictions, batch_company, batch_date, runName)

        return batch_trueLabels, batch_predictions

    @staticmethod
    def PrintResults(trueLabels, predictions, companies, dates, runName):
        pd.DataFrame({"True Labels": trueLabels,
                      "Predictions": predictions,
                      "Companies": companies,
                      "Dates": dates}).to_csv(
            f'{const.TrainedModelDirectory}'
            f'{runName}/test_result.csv',
            index=False)


class AnalyzerType(enum.Enum):
    Classification = 1
    Linear = 2
