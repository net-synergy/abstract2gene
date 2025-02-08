import json
import os

from abstract2gene.dataset import bioc2dataset, dataset_path

dataset = bioc2dataset(list(range(10)), max_cpu=16)

save_path = dataset_path("bioc")
if os.path.isdir(save_path):
    for f in os.listdir():
        os.unlink(f)
    os.rmdir(save_path)

dataset.save_to_disk(save_path, max_shard_size="250MB")
