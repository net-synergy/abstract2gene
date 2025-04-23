"""Remove the gene collection from the qdrant database."""

from webapp import config as cfg
from webapp import database

client = database.connect()
if client.collection_exists(cfg.collection_name):
    client.delete_collection(cfg.collection_name)
