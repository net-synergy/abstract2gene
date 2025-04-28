"""Read the webapps config file."""

import tomllib

_conf_file = "a2g.toml"
try:
    with open(_conf_file, "rb") as fp:
        conf = tomllib.load(fp)
        experiment_conf = conf.get("experiments", {})
        conf = conf["site"]
except (KeyError, FileNotFoundError):
    conf = {}

labels_per_batch = 16
min_genes = 5
gene_thresh = 0.5
results_per_page = 20
use_auth = True
hf_user = "dconnell"

hf_user = experiment_conf.get("hf_user", hf_user)

if "engine" in conf:
    labels_per_batch = conf["engine"].get("labels_per_batch", labels_per_batch)

model_name = f"abstract2gene_lpb_{labels_per_batch}"

if "ui" in conf:
    ui = conf["ui"]
    min_genes = ui.get("min_genes_displayed", min_genes)
    gene_thresh = ui.get("gene_thresh", gene_thresh)
    results_per_page = ui.get("results_per_page", results_per_page)

if "auth" in conf:
    use_auth = conf["auth"].get("enabled", use_auth)

# Don't think these need to be modified by user
collection_name = f"gene_predictions_{model_name}"
tmp_collection_name = "user_predictions"
