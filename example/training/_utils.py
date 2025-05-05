import time
from typing import Callable, TypeAlias

import datasets
import numpy as np
from datasets import Dataset

import example._config as cfg
from abstract2gene.dataset import dataset_generator, mutators

rng: TypeAlias = Callable[[], int]


def make_seed_generator(seed: int) -> rng:
    _seed_generator = np.random.default_rng(seed=seed).integers

    def seed_generator() -> int:
        return int(_seed_generator(0, 9999, size=(1,))[0])

    return seed_generator


def load_dataset(
    files: list[str],
    batch_size: int,
    n_batches: int,
    mask: str | list[str] | None,
    labels: str | list[str],
    seed_generator: rng,
    permute_prob: float = 0,
    augment_label: float = 0,
    attempt: int = 0,
) -> dict[str, Dataset]:
    """Generate a dataset with n_batches of batch_size from a list of files.

    The result will be a dictionary of datasets. One key per label used. The
    dataset is in the form of anchor--positive pairs.

    Parameters
    ----------
    files : list[str]
        The list of pubtator3_abstracts files to use (from huggingface dataset
        hub).
    batch_size : int
        Number of abstracts per batch.
    n_batches : int
        Number of unique batches.
    mask : str, list[str], or None
        If not None mask the given annotations (example "gene").
    labels : str | list[str]
        Which annotations should be used as labels. One dataset will be created
        per label and the result will be a dictionary where the label name is
        the key.
    seed_generator: rng
        A function that returns a random seed when it is called. If the
        seed_generator is reproducible it can be used to ensure the results are
        reproducible.
    permute_prob : float (default 0)
        The proportion of annotations that should be permuted instead of
        masked. A permuted annotation is replaced by a randomly selected
        annatotion from elsewhere in the dataset. (Only used using `mask`).
    augment_label : float (default 0)
        The proportion of all abstracts that should come from augmented labels.
        Augmented labels are labels given to an abstract that does not have
        PubTator3 annotations based on the citation network. If the publication
        is cited by an unexpectedly high proportion of publications with a
        given annotation that will be given as an annotation.

    """
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

    if mask is not None:
        dataset = mutators.mask_abstract(
            dataset,
            mask,
            permute_prob,
            seed=seed_generator(),
            max_cpu=cfg.max_cpu,
        )

    if isinstance(labels, str):
        labels = [labels]

    return {
        lab: dataset_generator(
            dataset,
            seed=seed_generator(),
            batch_size=batch_size,
            n_batches=n_batches,
            augment_label=augment_label,
        )
        for lab in labels
    }
