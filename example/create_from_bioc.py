import json
import os

from abstract2gene.dataset import bioc2dataset, dataset_path

dataset = bioc2dataset(list(range(10)), max_cpu=16)

save_path = dataset_path("bioc")
if os.path.isdir(save_path):
    for f in os.listdir():
        os.unlink(f)
    os.rmdir(save_path)

dataset = mutators.attach_pubmed_genes(dataset, "gene2pubmed", max_cpu=1)


def clear_save(path):
    if not os.path.isdir(path):
        return

    print(path)
    for f in os.listdir(path):
        abs_f = os.path.join(path, f)
        print(f, abs_f)
        if os.path.isdir(abs_f):
            clear_save(abs_f)
        else:
            os.unlink(abs_f)

    os.rmdir(path)


clear_save(save_path)
dataset.save_to_disk(save_path, max_shard_size="250MB")
