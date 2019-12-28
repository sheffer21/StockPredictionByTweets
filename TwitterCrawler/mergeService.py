import Common.Constants as c
import pandas as pd
import os


class MergeService:

    @staticmethod
    def merge(logger):

        logger.printAndLog(c.MessageType.Regular, "Merging database files...")
        dirname = os.path.dirname(__file__)
        mergePath = f'{c.twitterCrawlerDataBaseDir}{c.twitterCrawlerMergedFilesName}'
        mergePath = os.path.join(dirname, mergePath)

        old_merge_count = 0
        if os.path.exists(mergePath):
            old_merge = pd.read_csv(mergePath)
            old_merge_count = old_merge.shape[0]

        result = pd.DataFrame()

        dirPath = os.path.join(dirname, c.twitterCrawlerDataBaseDir)

        # Load Tweets from all files
        for file in os.listdir(dirPath):
            if file == c.twitterCrawlerMergedFilesName:
                break

            path = os.path.join(dirPath, file)
            logger.printAndLog(c.MessageType.Regular, f"Loading file: {path}")
            result = pd.concat([result, pd.read_csv(path)], sort=True)

        # Get all the unique Tweets
        unique_result = result.drop_duplicates()
        unique_result.to_csv(mergePath, index=False)
        logger.printAndLog(c.MessageType.Regular,
                           f"Added additional {unique_result.shape[0] - old_merge_count} to merge file")
        logger.printAndLog(c.MessageType.Success, "Successfully created a final merged database of Tweets")
