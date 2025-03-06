"""Upload the PubTator3 abstracts dataset to HuggingFace.

Parses the entire PubTator3 BioCXML archive set and converts them to a
HuggingFace DatasetDict. Each of the original BioCXML files gets its own split
in the dataset.

Results are pushed to the HuggingFace Hub.
"""

import os
import shutil

import datasets
from huggingface_hub import upload_large_folder
from huggingface_hub.repocard import DatasetCard, DatasetCardData

from abstract2gene.data import dataset_path
from abstract2gene.dataset import bioc2dataset, mutators


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


save_path = dataset_path("bioc")

dataset = bioc2dataset(range(10), max_cpu=60)
dataset = mutators.attach_references(dataset)

dataset.save_to_disk(save_path, max_shard_size="250MB")

os.mkdir(os.path.join(save_path, "data"))
for k in dataset:
    source = os.path.join(save_path, k)
    dest = os.path.join(save_path, "data", k)
    shutil.move(source, dest)

upload_large_folder(
    "dconnell/pubtator3_abstracts",
    folder_path=save_path,
    repo_type="dataset",
    num_workers=20,
)

for k in dataset:
    source = os.path.join(save_path, "data", k)
    dest = os.path.join(save_path, k)
    shutil.move(source, dest)
os.rmdir(os.path.join(save_path, "data"))


card_data = DatasetCardData(language="en")
card = DatasetCard.from_template(
    card_data, template_path="abstract2gene/dataset/README.md"
)

card.push_to_hub("dconnell/pubtator3_abstracts")
