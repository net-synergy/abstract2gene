"""Read the webapps config file."""

import os
import tomllib

_conf_file = "site.toml"
if not os.path.exists(_conf_file):
    conf = {}
else:
    with open(_conf_file, "rb") as fp:
        conf = tomllib.load(fp)

model_name = "a2g_768dim_per_batch_16"
min_genes = 5
gene_thresh = 0.5
results_per_page = 20
algorithm = "HS256"
access_token_expire_time = 3600 * 24

if "engine" in conf:
    model_name = conf["engine"].get("model_name", model_name)

if "ui" in conf:
    ui = conf["ui"]
    min_genes = ui.get("min_genes_displayed", min_genes)
    gene_thresh = ui.get("gene_thresh", gene_thresh)
    results_per_page = ui.get("results_per_page", results_per_page)

# Don't think these need to be modified by user
collection_name = f"gene_predictions_{model_name}"
tmp_collection_name = "user_predictions"
