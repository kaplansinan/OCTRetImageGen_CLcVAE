import json
from datetime import datetime
import os
import sys
import logging
from shutil import copyfile
sys.path.append('../')
from src.file_utils import exists_or_mkdir

class Logger:
    """
    Logger instance creates a training log folder named by datetime.
    Has utility functions for creating or appending logs.
    Example:
    # create logger instance
    logger = Logger(root='./output/', log_level=logging.DEBUG)
    # log config file to logdir
    logger.log_file("./config.py")
    # log event to train.log file in logdir. Prints also to console depending on log level.
    logger.log(level=logging.INFO, msg="Data loaded ok")
    """

    def __init__(self,
        root="./output/",
        log_level=logging.DEBUG,
        name=''
        ):
        """
        Init logger
        Arguments:
            root      (str): root folder where logdirs are created
            log_level (int): logging level that determines what is printed to console, default=logging.DEBUG
            name      (str): optional name for training (appended to logdir)
        Returns
            logger (Logger): Logger instance
        """
        # create logdir
        self.logdir = self._create_output_folder(root=root, name=name)
        # set logfile
        logging.basicConfig(
            filename=os.path.join(self.logdir, 'train.log'),
            level=log_level
            )

    def _create_output_folder(self, root="./output/", name=""):
        """
        Creates datetime named logdir to root folder
        Arguments:
            root   (str): root folder where logdirs are created
            name   (str): optional name for training (appended to logdir)
        Returns
            logdir (str): Path to created logdir
        """
        logdir = os.path.join(root, datetime.now().strftime("%Y%m%d-%H%M%S") + ('' if name=='' else '-') + name)
        exists_or_mkdir(root)
        exists_or_mkdir(logdir)
        return logdir

    def log_file(self, fn):
        """
        Copy a file to logdir.
        Arguments:
            fn (str): file to copy inside logdir
        """
        copyfile(fn, os.path.join(self.logdir, os.path.basename(fn)))

    def log_dict(self, dictionary, fn='parameters.json'):
        """
        Log python dictionary as json file.
        Arguments:
            dictionary (dict): dictionary to log
            fn          (str): file name to save to, default="parameters.json"
        """
        with open(os.path.join(self.logdir,fn), 'w') as text_file:
            text_file.write(json.dumps(dictionary, indent=4))

    def log(self, level:int=logging.INFO, msg:str=""):
        """
        Log events. A wrapper for logging module's log function.
        Arguments:
            level (int): logging level, default=logging.INFO
            msg   (str): log message
        """
        logging.log(level=level, msg=msg)
