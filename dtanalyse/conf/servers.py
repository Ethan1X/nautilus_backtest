#encoding=utf-8

from pymongo import MongoClient

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conf.servers_shift import SERVERS_CONF


def cnn_mongo_client(server_name):
    return MongoClient(SERVERS_CONF[server_name]['mongo_uri']+'/?socketTimeoutMS=600000', connect=False)

def cnn_mongo_client_aws(server_name):
    return MongoClient(SERVERS_CONF[server_name]['acc_str'], connect=False) 