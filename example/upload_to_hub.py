"""Upload encoder and dataset to Hugging Face Hub.

Requires being logged in as the Hugging Face user set in the config. If not set
this is "dconnell".
"""

import os
import shutil

import datasets
import huggingface_hub as hf_hub
import sentence_transformers
from huggingface_hub.repocard import DatasetCard, DatasetCardData

import example._config as cfg
from abstract2gene.data import dataset_path, encoder_path

## Upload dataset
save_path = dataset_path("bioc")

dataset = datasets.load_from_disk(save_path)

os.mkdir(os.path.join(save_path, "data"))
for k in dataset:
    source = os.path.join(save_path, k)
    dest = os.path.join(save_path, "data", k)
    shutil.move(source, dest)

hf_hub.upload_large_folder(
    f"{cfg.hf_user}/pubtator3_abstracts",
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

card.push_to_hub(f"{cfg.hf_user}/pubtator3_abstracts")

## Upload encoder
name = "PubMedNCL-abstract2gene"
encoder = sentence_transformers.SentenceTransformer(encoder_path(name))
encoder.push_to_hub(
    f"{cfg.hf_user}/{name}",
    private=True,
    local_model_path=encoder_path(name),
    train_datasets=[f"{cfg.hf_user}/pubtator3_abstracts"],
    exist_ok=True,
)
