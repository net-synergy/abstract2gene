import os
import shutil

import datasets
from huggingface_hub.repocard import DatasetCard, DatasetCardData

from abstract2gene.dataset import bioc2dataset, dataset_path, mutators


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


save_path = dataset_path("bioc_checkpoint")
clear_save(save_path)
dataset = bioc2dataset(range(10), max_cpu=60)
dataset.save_to_disk(save_path, max_shard_size="250MB")

# dataset = datasets.load_from_disk(save_path)
dataset = mutators.attach_references(dataset)

clear_save(save_path)
save_path = dataset_path("bioc")
dataset.save_to_disk(save_path, max_shard_size="250MB")

dataset.push_to_hub(
    "dconnell/pubtator3_abstracts",
)

card_data = DatasetCardData(language="en")
card = DatasetCard.from_template(
    card_data, template_path="abstract2gene/dataset/README.md"
)

card.push_to_hub("dconnell/pubtator3_abstracts")
