import json
import os

from abstract2gene.dataset import bioc2dataset, dataset_path

dataset = bioc2dataset([2], max_cpu=4)

save_path = dataset_path("bioc_small")
if os.path.isdir(save_path):
    for f in os.listdir():
        os.unlink(f)
    os.rmdir(save_path)

dataset.save_to_disk(save_path, max_shard_size="1GB")

# FIXME: Should be a proper part of dataset and get saved by `save_to_disk`.
with open(os.path.join(save_path, "symbols.json"), "w") as f:
    json.dump(dataset.features["gene2pubtator"].feature.symbols, f)
