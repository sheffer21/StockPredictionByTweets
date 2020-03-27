import seaborn as sns
import matplotlib.pyplot as plt
import os
import common.constants as const
import numpy as np


def Plot_Training_Loss(loss_values, runName):
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
    Save_Plot(const.MachineLearnerStatisticsFolder, const.MachineLearnerTrainPlot, runName)


def Plot_Distributions(list, runName):
    # seaborn histogram
    sns.distplot(list, hist=True, kde=True,
                 bins=1000, color='blue',
                 hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title('Histogram of Followers')
    plt.xlabel('Followers')
    plt.ylabel('#Tweets')

    Save_Plot(const.MachineLearnerStatisticsFolder, const.MachineLearnerFollowersPlot, runName)


def Plot_DataBase_Labels_Statistics(labels, labels_names, runName):
    xCoordinates = np.arange(len(labels_names))
    heights = [sum([1 for label_value in labels if label_value == label_index])
               for label_index in range(len(labels_names))]

    fig, ax = plt.subplots()
    ax.set_ylabel('Number of Tweets')
    ax.set_title('Number of Tweets by label')
    ax.set_xticks(xCoordinates)
    ax.set_xticklabels(labels_names)
    ax.bar(x=xCoordinates, height=heights, width=0.40, color=['green'])

    # Arrange labels
    for item in (ax.get_xticklabels()):
        item.set_fontsize(9)

    Save_Plot(const.MachineLearnerStatisticsFolder, const.MachineLearnerLabelsPlot, runName)


def Save_Plot(directory, fileName, runName):
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 10)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}/{fileName}_{runName}.png", dpi=700)
    plt.clf()
