"""Populate qdrant database with abstract data.

This requires qdrant to be running and the model that will be used to predict
the abstract's genes.

This is required to be run before the webapp can be started.

Currently is intended to create a new database collection. In the future, it
may make sense to allow adding new publications to an already created database.
"""

import json
import os
import sys

import datasets

import abstract2gene as a2g
import webapp.config as cfg
from abstract2gene.data import model_path
from webapp import database

client = database.connect()
if client.collection_exists(cfg.collection_name):
    # Collection already exists; nothing to do so gracefully end script.
    sys.exit()

model = a2g.model.load_from_disk(cfg.model_name)
database.init_db(client, model, cfg.collection_name)

# TEMP: Starting with small subset, add full dataset later.
_file_template = "data/BioCXML_{archive}/data-{f_idx:05}-of-{f_total:05}.arrow"
data_files = _file_template.format(archive=0, f_idx=0, f_total=20)

dataset = datasets.load_dataset(
    "dconnell/pubtator3_abstracts", data_files=data_files
)["train"]
dataset = dataset.select(range(1000))

genes = {
    "symbol": a2g.dataset.mutators.get_gene_symbols(dataset),
    "entrez_id": dataset.features["gene"].feature.names,
}

if model.templates:
    genes = {
        k: [v[i] for i in model.templates.indices] for k, v in genes.items()
    }

with open(os.path.join(model_path(cfg.model_name), "genes.json"), "w") as js:
    json.dump(genes, js)

database.store_publications(client, dataset, model, cfg.collection_name)
