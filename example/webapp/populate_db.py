"""Populate qdrant database with abstract data.

This requires qdrant to be running and the model that will be used to predict
the abstract's genes.

This is required to be run before the webapp can be started.

Currently is intended to create a new database collection. In the future, it
may make sense to allow adding new publications to an already created database.
"""

import asyncio
import json
import os
import sys

import datasets

import abstract2gene as a2g
import webapp.config as cfg
from abstract2gene.data import model_path
from webapp import database


async def main():
    client = database.connect()
    if await client.collection_exists(cfg.collection_name):
        # Collection already exists; nothing to do so gracefully end script.
        sys.exit()

    model = a2g.model.load_from_disk(cfg.model_name)
    await database.init_db(client, model, cfg.collection_name)

    dataset = datasets.load_dataset(f"{cfg.hf_user}/pubtator3_abstracts")[
        "train"
    ]

    dataset = a2g.dataset.mutators.translate_to_human_orthologs(dataset)
    genes = {
        "symbol": a2g.dataset.mutators.get_gene_symbols(dataset),
        "entrez_id": dataset.features["gene"].feature.names,
    }

    if model.templates:
        indices = model.sync_indices(dataset)
        genes = {
            k: [v[int(i)] for i in indices if i > 0] for k, v in genes.items()
        }

    with open(
        os.path.join(model_path(cfg.model_name), "genes.json"), "w"
    ) as js:
        json.dump(genes, js)

    await database.store_publications(
        client, dataset, model, cfg.collection_name
    )


asyncio.run(main())
