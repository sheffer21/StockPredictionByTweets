import constants as c
from logger import Logger as Log
import pandas as pd
import os


def merge():

    Log.print_and_log(c.MessageType.Regular.value, "Merging database files...")
    merge_path = f'{c.twitterCrawlerDataBaseDir}{c.twitterCrawlerMergedFilesName}'

    old_merge_count = 0
    if os.path.exists(merge_path):
        old_merge = pd.read_csv(merge_path)
        old_merge_count = old_merge.shape[0]

    result = pd.DataFrame()

    # Load tweets from all files
    for file in os.listdir(c.twitterCrawlerDataBaseDir):
        if file == c.twitterCrawlerMergedFilesName:
            break

        path = os.path.join(c.twitterCrawlerDataBaseDir, file)
        Log.print_and_log(c.MessageType.Regular.value, f"Loading file: {path}")
        result = pd.concat([result, pd.read_csv(path)], sort=True)

    # Get all the unique tweets
    unique_result = result.drop_duplicates()
    unique_result.to_csv(merge_path, index=False)
    Log.print_and_log(c.MessageType.Regular.value,
                      f"Added additional {unique_result.shape[0] - old_merge_count} to merge file")
    Log.print_and_log(c.MessageType.Success.value, "Successfully created a final merged database of tweets")
