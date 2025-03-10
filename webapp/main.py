import datasets

import abstract2gene as a2g
from webapp import database

collection_name = "gene_predictions"
model = a2g.model.load_from_disk("a2g_768dim_per_batch_8")

client = database.connect()
if not client.collection_exists(collection_name):
    database.init_db(client, model, collection_name)

    # TEMP: Starting with small subset, add full dataset later.
    _file_template = (
        "data/BioCXML_{archive}/data-{f_idx:05}-of-{f_total:05}.arrow"
    )
    data_files = _file_template.format(archive=0, f_idx=0, f_total=20)

    dataset = datasets.load_dataset(
        "dconnell/pubtator3_abstracts", data_files=data_files
    )["train"]
    dataset = dataset.select(range(200))

    database.store_publications(client, dataset, model, collection_name)
    del dataset
