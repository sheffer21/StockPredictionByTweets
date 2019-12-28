from datetime import datetime
import os
import sys
import Common.Constants as const


class Logger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    internalLogFileDirectory = const.logsDirectory
    internalLogFileName = "{}_{}.log".format(const.logNamePrefix, datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    internalLogFilePath = "{}/{}".format(internalLogFileDirectory, internalLogFileName)

    def __init__(self):
        self.fileDirectory = Logger.internalLogFileDirectory
        self.fileName = Logger.internalLogFileName
        self.filePath = Logger.internalLogFilePath

    def createLogFileIfNotExists(self):
        if not os.path.isdir(self.fileDirectory):
            os.mkdir(self.fileDirectory)

    def printMessage(self, messageType, message):
        self.createLogFileIfNotExists()

        prefix = {
            const.MessageType.Regular: "",
            const.MessageType.Error: Logger.FAIL,
            const.MessageType.Success: Logger.OKGREEN,
            const.MessageType.Summarize: Logger.OKBLUE,
            const.MessageType.Header: Logger.HEADER
        }

        suffix = {
            const.MessageType.Regular: "",
            const.MessageType.Error: Logger.ENDC,
            const.MessageType.Success: Logger.ENDC,
            const.MessageType.Summarize: Logger.ENDC,
            const.MessageType.Header: Logger.ENDC
        }

        if messageType == "printLog":
            print(prefix.get(const.MessageType.Regular, "") +
                  "Log path: {}\n".format(self.filePath) +
                  suffix.get(messageType, ""))
        else:
            print(prefix.get(messageType, "") + message + suffix.get(messageType, ""))

    def printAndLog(self, messageType, message):

        self.createLogFileIfNotExists()

        # Print to console:
        self.printMessage(messageType, message)

        # Print to log:
        old_stdout = sys.stdout
        log_file = open(self.filePath, "a")
        sys.stdout = log_file
        self.printMessage(messageType, message)
        log_file.close()
        sys.stdout = old_stdout
