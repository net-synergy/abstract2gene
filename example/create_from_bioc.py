"""Upload the PubTator3 abstracts dataset to HuggingFace.

Parses the entire PubTator3 BioCXML archive set and converts them to a
HuggingFace DatasetDict. Each of the original BioCXML files gets its own split
in the dataset.

Results are pushed to the HuggingFace Hub.
"""

import argparse
import os

from abstract2gene.data import dataset_path
from abstract2gene.dataset import bioc2dataset, mutators
from example import config as cfg


def clear_save(path):
    if not os.path.isdir(path):
        return

    for f in os.listdir(path):
        abs_f = os.path.join(path, f)
        if os.path.isdir(abs_f):
            clear_save(abs_f)
        else:
            os.unlink(abs_f)

    os.rmdir(path)


n_cpu = cfg.max_cpu
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n_cpu",
        type=int,
        requried=False,
        default=cfg.max_cpu,
        help="Number of CPU processes to use to parse the Bioc files",
    )
    args = parser.parse_args()
    n_cpu = args.n_cpu

save_path = dataset_path("bioc")

dataset = bioc2dataset(range(10), max_cpu=n_cpu)
dataset = mutators.attach_references(dataset)

dataset.save_to_disk(save_path, max_shard_size="250MB")
