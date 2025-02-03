import json
import os

from abstract2gene.dataset import bioc2dataset, dataset_path

dataset = bioc2dataset(list(range(10)), max_cpu=16)

save_path = dataset_path("bioc")
if os.path.isdir(save_path):
    for f in os.listdir():
        os.unlink(f)
    os.rmdir(save_path)

dataset.save_to_disk(save_path, max_shard_size="1GB")

# FIXME: Should be a proper part of dataset and get saved by `save_to_disk`.
key = list(dataset.keys())[0]
with open(os.path.join(save_path, "symbols.json"), "w") as syms:
    json.dump(dataset[key].features["gene2pubtator"].feature.symbols, syms)
