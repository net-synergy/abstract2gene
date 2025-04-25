"""Upload encoder and dataset to Hugging Face Hub.

Requires being logged in as the Hugging Face user set in the config. If not set
this is "dconnell".
"""

import argparse
import os
import shutil

import datasets
import huggingface_hub as hf_hub
import sentence_transformers
from huggingface_hub.repocard import DatasetCard, DatasetCardData

import example._config as cfg
from abstract2gene.data import dataset_path, encoder_path

upload_dataset = True
upload_model = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upload_dataset",
        action="store_true",
        help="Whether to upload the dataset",
    )
    parser.add_argument(
        "--upload_model",
        action="store_true",
        help="Whether to upload the model",
    )
    args = parser.parse_args()
    upload_dataset = args.upload_dataset
    upload_model = args.upload_model


## Upload dataset
if upload_dataset:
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
if upload_model:
    local_name = cfg.encoder
    base_name, _, project_name = local_name.split("-")
    remote_name = "-".join((base_name, project_name))
    encoder = sentence_transformers.SentenceTransformer(
        encoder_path(local_name)
    )
    encoder.push_to_hub(
        f"{cfg.hf_user}/{remote_name}",
        private=True,
        local_model_path=encoder_path(local_name),
        train_datasets=[f"{cfg.hf_user}/pubtator3_abstracts"],
        exist_ok=True,
    )
