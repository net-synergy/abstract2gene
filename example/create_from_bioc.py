import os

from abstract2gene.data import default_data_dir
from abstract2gene.dataset import bioc2dataset

dataset = bioc2dataset([1], max_cpu=4)

save_path = os.path.join(default_data_dir("datasets"), "bioc")
if os.path.isdir(save_path):
    for f in os.listdir():
        os.unlink(f)
    os.rmdir(save_path)

dataset.save_to_disk(save_path, max_shard_size="1GB")
