import seaborn as sns
import matplotlib.pyplot as plt
import os
import common.constants as const


def Plot_Training_Loss(loss_values):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o')

    # Label the plot.
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # plt.show()
    Save_Plot(const.MachineLearnerStatisticsFolder, const.MachineLearnerTrainPlot)


def Plot_Training_Labels(self):
    companiesKeywords, companiesPossibleKeywords = self.GetCompaniesKeywordsDataCount()
    labels = list(key.split(" ", 2)[0] for key in companiesKeywords.keys())
    xCoordinates = np.arange(len(labels))
    keywordsHeights = companiesKeywords.values()
    possibleKeywordsHeights = companiesPossibleKeywords.values()

    fig, ax = plt.subplots()
    width = 0.35
    rect1 = ax.bar(xCoordinates - width / 2, keywordsHeights, width, label=const.COMPANY_KEYWORDS_COLUMN)
    rect2 = ax.bar(xCoordinates + width / 2, possibleKeywordsHeights, width,
                   label=const.COMPANY_POSSIBLE_KEYWORDS_COLUMN)

    ax.set_ylabel('Tweets')
    ax.set_title('Number of Tweets by Company')
    ax.set_xticks(xCoordinates)
    ax.set_xticklabels(labels)
    ax.legend()

    DataBaseStatistics.autoLabel(rect1, ax)
    DataBaseStatistics.autoLabel(rect2, ax)

    # Arrange labels
    for item in (ax.get_xticklabels()):
        item.set_fontsize(9)

    self.SavePlotToFile(const.twitterCrawlerPossibleKeywordsStatistics)


def Save_Plot(self, directory, fileName, runName):
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 10)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}/{fileName}_{runName}.png", dpi=700)