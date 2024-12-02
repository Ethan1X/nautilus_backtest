#coding=utf-8
#author:shiyuming hudongdong 
'''
统一的log方法
'''

import logging, os, sys, time, datetime, traceback
import logging.handlers

def initlog(path, log_filename, log_level=logging.INFO, interval=2, backup_count = 24*2):
    logger = logging.getLogger()
    LOG_FILE = "%s/%s" % (path, log_filename)
    print(LOG_FILE)
    handler = None
    if path is None or log_filename is None:
        #试用std.out
        handler = logging.StreamHandler(stream=sys.stdout)
    else:
        handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE, when='H', interval=interval, backupCount=backup_count)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(filename)s %(funcName)s %(lineno)s %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger
