from datetime import datetime
import os
import sys


class Logger:

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    logFilePath = "logs/"
    logFileName = "messages_{}.log".format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    @staticmethod
    def create_file_if_not_exists():
        if not os.path.isdir(Logger.logFilePath):
            os.mkdir(Logger.logFilePath)

    @staticmethod
    def print_message(message_type, message):
        Logger.create_file_if_not_exists()

        prefix = {
            "Regular": "",
            "Error": Logger.FAIL,
            "Success": Logger.OKGREEN,
            "Summarize": Logger.OKBLUE,
            "Header": Logger.HEADER
        }

        suffix = {
            "Regular": "",
            "Error": Logger.ENDC,
            "Success": Logger.ENDC,
            "Summarize": Logger.ENDC,
            "Header": Logger.ENDC
        }

        if message_type == "printLog":
            print(prefix.get("Regular", "") +
                  "Log path: {}{}\n".format(Logger.logFilePath, Logger.logFileName) +
                  suffix.get(message_type, ""))
        else:
            print(prefix.get(message_type, "") + message + suffix.get(message_type, ""))

    @staticmethod
    def print_and_log(message_type, message):

        Logger.create_file_if_not_exists()

        # Print to console:
        Logger.print_message(message_type, message)

        # Print to log:
        old_stdout = sys.stdout
        log_file = open(Logger.logFilePath + Logger.logFileName, "a")
        sys.stdout = log_file
        Logger.print_message(message_type, message)
        log_file.close()
        sys.stdout = old_stdout
