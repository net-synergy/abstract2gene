import time
from typing import Callable, TypeAlias

import datasets
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer

import example._config as cfg
from abstract2gene.dataset import dataset_generator, mutators
from abstract2gene.model import Model

rng: TypeAlias = Callable[[], int]


def make_seed_generator(seed: int) -> rng:
    _seed_generator = np.random.default_rng(seed=seed).integers

    def seed_generator() -> int:
        return int(_seed_generator(0, 9999, size=(1,))[0])

    return seed_generator


def load_dataset(
    files: list[str],
    model: Model | SentenceTransformer,
    batch_size: int,
    n_batches: int,
    mask: str | list[str] | None,
    labels: str | list[str],
    seed_generator: rng,
    permute_prob: float = 0,
    attempt: int = 0,
) -> dict[str, Dataset]:
    try:
        dataset = datasets.load_dataset(
            f"{cfg.hf_user}/pubtator3_abstracts", data_files=files
        )["train"]
    except ValueError as err:
        # When it can't find the dataset on the HF Hub for some reason.
        if attempt > 3:
            raise err

        time.sleep(5)
        return load_dataset(
            files,
            model,
            batch_size,
            n_batches,
            mask,
            labels,
            seed_generator,
            permute_prob,
            attempt + 1,
        )

    dataset = mutators.translate_to_human_orthologs(
        dataset, max_cpu=cfg.max_cpu
    )

    if isinstance(model, Model):
        mask_token = model.mask_token
        sep_token = model.sep_token
    else:
        token_map = model.tokenizer.special_tokens_map
        mask_token = token_map["mask_token"]
        sep_token = token_map["sep_token"]

    if mask is not None:
        dataset = mutators.mask_abstract(
            dataset,
            mask,
            permute_prob,
            seed=seed_generator(),
            max_cpu=cfg.max_cpu,
            mask_token=mask_token,
        )

    if isinstance(labels, str):
        labels = [labels]

    return {
        lab: dataset_generator(
            dataset,
            seed=seed_generator(),
            batch_size=batch_size,
            n_batches=n_batches,
            sep_token=sep_token,
        )
        for lab in labels
    }
