"""A streamble dataset for on the fly generation of sequence triplets.

Training methods need a dataset with keys: "Anchor", "Positive", "Negative" but
the original dataset is not in this form and putting it in this form would
require coupling abstracts together.

To get better dataset flexibility, instead we use a generator to temporarily
pair two (or three) abstracts together for the duration of a batch. Every batch
will get a newly generated set of triplets.
"""

__ALL__ = ["dataset_generator"]

from typing import Any, Iterable

import numpy as np
import scipy as sp
from datasets import Dataset, Features, IterableDataset, Value


def dataset_generator(
    dataset: Dataset,
    max_labels: int = 10,
    seed: int = 0,
    batch_size: int = 32,
) -> IterableDataset:
    """Create a streamable dataset from another dataset.

    Generate (anchor, positive, label) triplets on the fly from another
    dataset.

    Max labels is the maximum number of labels an individual sample can be
    given. As the number of samples increase it losses specificity for that
    label and can't be seen as a positive example.

    Additionally unlabeled data is dropped since it cannot have a positive
    example without a label.

    """
    labels = _lol_to_csc(dataset["gene2pubtator"])
    samples = dataset["abstract"]

    overlabeled = labels.sum(axis=1) > max_labels
    unlabeled = labels.sum(axis=1) == 0
    sample_mask = np.logical_not(np.logical_or(unlabeled, overlabeled))
    samples = [sample for mask, sample in zip(sample_mask, samples) if mask]
    labels = labels[sample_mask, :]
    labels = labels[:, labels.sum(axis=0) > 2]

    if batch_size > labels.shape[1]:
        RuntimeError(
            "Not enough unique labels to generate a full batch."
            + f" Lower batch size to less than {labels.shape[1]}"
        )

    def _generator() -> Iterable[dict[str, Any]]:
        rng = np.random.default_rng(seed=seed)
        n_labels = labels.shape[1]

        while True:
            mix_idx = rng.permuted(np.arange(len(samples)))
            batch_samps = [samples[i] for i in mix_idx]
            batch_labels = labels[mix_idx, :]
            for i, label in enumerate(rng.choice(n_labels, batch_size)):
                label_mask = batch_labels[:, [label]].toarray().squeeze()
                positive_idx = np.arange(len(batch_samps))[label_mask]
                negative_idx = np.arange(len(batch_samps))[
                    np.logical_not(label_mask)
                ]

                yield {
                    "anchor": batch_samps[positive_idx[0]],
                    "positive": batch_samps[positive_idx[1]],
                    "negative": batch_samps[
                        negative_idx[i % np.logical_not(label_mask).sum()]
                    ],
                }

    feats = Features(
        {k: Value(dtype="string") for k in ["anchor", "positive", "negative"]}
    )
    return IterableDataset.from_generator(_generator, features=feats)


def _lol_to_csc(lists: list[list[int]]) -> sp.sparse.sparray:
    nrows = len(lists)
    ncols = max((col for ls in lists for col in ls)) + 1
    numel = sum((len(ls) for ls in lists))
    coords = np.zeros((2, numel))

    count = 0
    for i, ls in enumerate(lists):
        for el in ls:
            coords[:, count] = [i, el]
            count += 1

    return sp.sparse.coo_array(
        (np.ones((numel,), dtype=np.bool), coords), shape=(nrows, ncols)
    ).tocsc()
