"""
pymongo 3.12.0
"""
from pymongo import MongoClient
import pymongo

class MongoDB:
    """
    This class is a tool to connect mongodb
    TODO need test before create
    """
    def __init__(self, host, user, passwd, port):
        """ initialize args"""
        self.host, self.user, self.passwd, self.port = host, user, passwd, port

    def connect(self, db, collection):
        """
        connect to specific collection in specific database
        @param
        db: database
        collection: collection
        @return
        mongo_collection: connection to the collection
        """
        mongo_client = pymongo.MongoClient(self.host, self.port)
        mongo_auth = mongo_client.admin
        mongo_auth.authenticate(self.user, self.passwd)
        mongo_db = mongo_client[db] 
        mongo_collection = mongo_db[collection]

        return mongo_collection