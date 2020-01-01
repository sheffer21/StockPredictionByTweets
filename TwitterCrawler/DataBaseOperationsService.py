import pandas as pd
import os
import constants as const


class DataBaseOperationsService:

    def __init__(self, logger):
        self.logger = logger

    def merge(self):
        self.logger.printAndLog(const.MessageType.Regular, "Merging database files...")
        mergeDataFilePath = DataBaseOperationsService.GetMergedDataBaseFilePath()

        old_merge_count = 0
        if os.path.exists(mergeDataFilePath):
            old_merge = pd.read_csv(mergeDataFilePath)
            old_merge_count = old_merge.shape[0]

        result = pd.DataFrame()

        # Load tweets from all files
        for file in self.GetAllDataBaseFilesPaths():
            self.logger.printAndLog(const.MessageType.Regular.value, f"Loading file: {os.path.basename(file)}")
            data = pd.read_csv(file)
            result = pd.concat([result, data], sort=False)

        # Get all the unique tweets
        unique_result = self.DropDataDuplicates(result, const.ID_COLUMN)
        self.SaveToCsv(unique_result, mergeDataFilePath)
        self.logger.printAndLog(const.MessageType.Regular.value,
                                f"Added additional {unique_result.shape[0] - old_merge_count} to merge file")
        self.logger.printAndLog(const.MessageType.Regular.value, f"Data base current size: {unique_result.shape[0]}")
        self.logger.printAndLog(const.MessageType.Success.value,
                                "Successfully created a final merged database of tweets")

        self.DeleteDataBaseLogExceptLast(10)

    def DeleteDataBaseLogExceptLast(self, num):
        files = self.GetAllDataBaseFilesPaths()
        if len(files) <= 10:
            return

        files.sort(reverse=True)
        data = pd.DataFrame()
        finalPath = ""
        filesToRemove = []
        for index, file in enumerate(files):
            if index < num - 1:
                continue

            if index == num - 1:
                finalPath = file

            filesToRemove.append(file)
            data = pd.concat([data, pd.read_csv(file)], sort=False)

        self.DeleteFiles(filesToRemove)
        unique_data = self.DropDataDuplicates(data, const.ID_COLUMN)
        self.SaveToCsv(unique_data, finalPath)
        self.logger.printAndLog(
            const.MessageType.Regular, f'Merge all deleted files data to last one of them {finalPath}')

    @staticmethod
    def DeleteFiles(self, files):
        for file in files:
            self.logger.printAndLog(const.MessageType.Regular, f'Delete file {file}')
            os.remove(file)

    def AddCompanyNameToFiles(self):
        symbol_to_company = DataBaseOperationsService.GetSymbolToCompanyDictionary()
        for file in DataBaseOperationsService.GetAllDataBaseFilesPaths():
            data = pd.read_csv(file)
            if const.COMPANY_COLUMN in data.columns:
                continue

            self.AddColumnToDataByReferenceColumnValue(data,
                                                       symbol_to_company,
                                                       const.STOCK_SYMBOL_COLUMN,
                                                       const.COMPANY_COLUMN)
            self.SaveToCsv(data, file)
            self.logger.printAndLog(const.MessageType.Regular, f"Add company name to file {os.path.basename(file)}")

    def AddFollowersCountToFiles(self, crawler):
        for file in DataBaseOperationsService.GetAllDataBaseFilesPaths():
            data = pd.read_csv(file)
            if const.USER_FOLLOWERS_COLUMN in data.columns:
                continue
            crawler.addUsersFollowers(tweets=data)
            DataBaseOperationsService.SaveToCsv(data=data, path=file)
            self.logger.printAndLog(const.MessageType.Regular, f"Add followers count to file {os.path.basename(file)}")

    @staticmethod
    def AddColumnToDataByReferenceColumnValue(data, referenceToValue, referenceColumn, destinationColumn):
        destinationValues = []
        for reference in data[referenceColumn].values:
            if reference in referenceToValue:
                destinationValues.append(referenceToValue[reference])
            else:
                destinationValues.append("")

        data[destinationColumn] = destinationValues

    @staticmethod
    def GetSymbolToCompanyDictionary():
        companies_data = pd.read_csv(DataBaseOperationsService.GetCompaniesFilePath())
        companies_dictionary = {}
        for company in companies_data.values:
            companies_dictionary[company[1]] = company[0]

        return companies_dictionary

    @staticmethod
    def GetAllDataBaseFilesPaths(include_merge_file=False):
        files = []
        for file in os.listdir(DataBaseOperationsService.GetDataBaseDir()):
            if (not include_merge_file) and file == const.twitterCrawlerMergedFilesName:
                continue
            files.append(os.path.join(DataBaseOperationsService.GetDataBaseDir(), file))

        return files

    @staticmethod
    def DropDataDuplicates(data, column):
        return data.drop_duplicates(column)

    @staticmethod
    def SaveToCsv(data, path):
        data.to_csv(path, index=False)

    @staticmethod
    def GetCompaniesFilePath():
        dirName = os.path.dirname(__file__)
        return os.path.join(dirName, const.companiesPath)

    @staticmethod
    def GetMergedDataBaseFilePath():
        dirName = DataBaseOperationsService.GetDataBaseDir()
        mergePath = f'{const.twitterCrawlerMergedFilesName}'
        return os.path.join(dirName, mergePath)

    @staticmethod
    def GetDataBaseDir():
        dirName = os.path.dirname(__file__)
        return os.path.join(dirName, const.twitterCrawlerDataBaseDir)
