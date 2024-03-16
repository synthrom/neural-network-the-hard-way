from os.path import expanduser, isdir, exists
from os import makedirs
import logging, logging.handlers

"""Creates a logging instance"""
def create_logger(path='./',name='script',loggerLevel='INFO',fileHandlerLevel='INFO',consoleHandlerLevel='INFO',formatter='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s ',maxBytes=10485760,backupCount=20):
    if isdir(path) is False:
        path = "./"
    logger = logging.getLogger("{}.log".format(name))
    try:
        logger.setLevel(getattr(logging,(loggerLevel.upper())))
    except:
        logger.setLevel(logging.DEBUG)
    if not exists("{}".format(path)):
        makedirs("{}".format(path))
    fileHandler = logging.handlers.RotatingFileHandler("{}/{}.log".format(path,name),maxBytes=maxBytes, backupCount=backupCount)
    consoleHandler = logging.StreamHandler()

    # set logging level to match level input
    try:
        fileHandler.setLevel(getattr(logging,(fileHandlerLevel.upper())))
    except:
        try:
            fileHandler.setLevel(int(fileHandlerLevel))
        except:
            fileHandler.setLevel(logging.DEBUG)
    try:
        consoleHandler.setLevel(getattr(logging,(consoleHandlerLevel.upper())))
    except:
        try:
            consoleHandler.setLevel(int(consoleHandlerLevel))
        except:
            consoleHandler.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(formatter)
    fileHandler.setFormatter(logger_formatter)
    consoleHandler.setFormatter(logger_formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    return logger
