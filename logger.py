import constants as c
import os
import sys


class Logger:

    if not os.path.isdir(c.logFilePath):
        os.mkdir(c.logFilePath)

    @staticmethod
    def print_message(message_type, message):
        prefix = {
            "Regular": "",
            "Error": c.FAIL,
            "Success": c.OKGREEN,
            "Summarize": c.OKBLUE,
            "Header": c.HEADER
        }

        suffix = {
            "Regular": "",
            "Error": c.ENDC,
            "Success": c.ENDC,
            "Summarize": c.ENDC,
            "Header": c.ENDC
        }

        if message_type == "printLog":
            print(prefix.get("Regular", "") +
                  "Log path: {}{}\n".format(c.logFilePath, c.logFileName) +
                  suffix.get(message_type, ""))
        else:
            print(prefix.get(message_type, "") + message + suffix.get(message_type, ""))

    @staticmethod
    def print_and_log(message_type, message):
        # Print to console:
        Logger.print_message(message_type, message)

        # Print to log:
        old_stdout = sys.stdout
        log_file = open(c.logFilePath + c.logFileName, "a")
        sys.stdout = log_file
        Logger.print_message(message_type, message)
        log_file.close()
        sys.stdout = old_stdout
